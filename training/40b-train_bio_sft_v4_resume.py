#!/usr/bin/env python
# coding: utf-8
"""
40-train_bio_sft_v4.py

Bio-SFT v4: 修复 SFT v3 的 chat tag collapse + Structure/Mutation 失败

改动:
  1. Prompt template: <User>/<Assistant> -> 纯 Alpaca (### Instruction/Answer)
  2. Loss masking: 只对 {output} 部分计 loss (instruction tuning 标准)
  3. MAX_LENGTH: 1024 -> 1536 (覆盖长 Structure 输出)
  4. 任务加权采样: Structure x3, Mutation x2, 其他 x1
  5. 从 v3 LoRA 继续训, 不从头开始

预计 12 GPU 小时, 单卡 H20.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME
from peft import LoraConfig, inject_adapter_in_model

# ================= 1. 配置 =================
BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
# Resume from checkpoint-1000 (interrupted at step 1000 by disk full)
# Load LoRA + embedding from checkpoint-1000 as warm start
INIT_DIR = "/root/autodl-tmp/dnagpt/checkpoints_sft_v4/checkpoint-1000"
TRAIN_FILE = "/root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train_with_remote.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v4-sft"
CHECKPOINT_DIR = "/root/autodl-fs/checkpoints_sft_v4_resume"

MAX_LENGTH = 1536
TASK_WEIGHTS = {"Structure": 3, "Mutation": 2}  # 其他默认 1


def task_of(instruction):
    instr_low = instruction.lower()
    if "3di" in instruction.lower() or "secondary structure" in instr_low or "dssp" in instr_low or "foldseek" in instr_low:
        return "Structure"
    if "mutation" in instr_low or "wild-type" in instr_low or "mutant" in instr_low:
        return "Mutation"
    return "Other"


# ================= 2. 加载 + 加权数据 =================
print("Loading and oversampling data...", flush=True)
records = []
with open(TRAIN_FILE) as f:
    for line in f:
        records.append(json.loads(line))

print(f"  Total: {len(records)}")
weighted = []
counts = {"Structure": 0, "Mutation": 0, "Other": 0}
for r in records:
    task = task_of(r.get("instruction", ""))
    counts[task] += 1
    weight = TASK_WEIGHTS.get(task, 1)
    for _ in range(weight):
        weighted.append(r)

print(f"  Original counts: {counts}")
print(f"  After oversampling: {len(weighted)}")

dataset = Dataset.from_list(weighted)


# ================= 3. 纯 Alpaca format (no chat tag) =================
def format_alpaca(instruction, input_text, output):
    """Alpaca 标准, 不带 <User>/<Assistant>."""
    if input_text and input_text.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"{input_text}\n\n"
            f"### Answer:\n",
            output
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Answer:\n",
        output
    )


# ================= 4. 加载模型 =================
print("Loading model (4-bit)...", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map={"": 0},
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v3-sft-remote")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 5. LoRA + load v3 =================
peft_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
for p in model.parameters():
    p.requires_grad = False
inject_adapter_in_model(peft_config, model.model.language_model, adapter_name="default")
model._hf_peft_config_loaded = True

print("Loading v3 LoRA + embedding...", flush=True)
v3_lora = torch.load(f"{INIT_DIR}/lora_weights.pt", map_location="cpu")
ms = model.state_dict()
loaded = 0
for k, v in v3_lora.items():
    if k in ms:
        ms[k].copy_(v); loaded += 1
print(f"  Loaded {loaded} v3 LoRA tensors")

v3_embed = torch.load(f"{INIT_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(v3_embed)

for p in model.get_input_embeddings().parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")


# ================= 6. Tokenize with loss masking =================
def tok_fn(examples):
    """关键: 把 prompt 部分的 label 设为 -100, 只对 output 计 loss."""
    input_ids_list = []
    labels_list = []
    am_list = []
    for i in range(len(examples["instruction"])):
        prompt, ans = format_alpaca(
            examples["instruction"][i],
            examples["input"][i],
            examples["output"][i],
        )
        ans_with_eos = ans + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH-256,
                                add_special_tokens=False).input_ids
        ans_ids = tokenizer(ans_with_eos, truncation=True, max_length=MAX_LENGTH-len(prompt_ids),
                             add_special_tokens=False).input_ids
        full_ids = prompt_ids + ans_ids
        # labels: prompt 部分 -100, answer 部分保留
        labels = [-100] * len(prompt_ids) + ans_ids[:]
        am = [1] * len(full_ids)
        input_ids_list.append(full_ids)
        labels_list.append(labels)
        am_list.append(am)
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": am_list,
    }


print("Tokenizing...", flush=True)
tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names, num_proc=8)


# ================= 7. Collator =================
class Collator:
    def __init__(self, tok, max_len):
        self.tok = tok
        self.max_len = max_len
        self.pad = tok.pad_token_id or 0
    def __call__(self, feats):
        ml = min(max(len(f["input_ids"]) for f in feats), self.max_len)
        ids, lbl, am = [], [], []
        for f in feats:
            x = f["input_ids"][:ml]
            y = f["labels"][:ml]
            a = f["attention_mask"][:ml]
            pl = ml - len(x)
            ids.append(x + [self.pad]*pl)
            lbl.append(y + [-100]*pl)
            am.append(a + [0]*pl)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lbl, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(len(feats), ml, dtype=torch.long),
        }

collator = Collator(tokenizer, MAX_LENGTH)


# ================= 8. LoRATrainer =================
class LoRATrainer(Trainer):
    def _save_lora(self, out_dir, model=None):
        m = model or self.model
        bm = m.module if hasattr(m, "module") else m
        os.makedirs(out_dir, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in bm.state_dict().items() if "lora_" in k}
        torch.save(sd, os.path.join(out_dir, "lora_weights.pt"))
        ew = bm.get_input_embeddings().weight.detach().cpu()
        torch.save(ew, os.path.join(out_dir, "embedding_weights.pt"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(out_dir)

    def save_model(self, output_dir=None, _internal_call=False):
        self._save_lora(output_dir or self.args.output_dir)

    def _save_checkpoint(self, model, trial):
        cf = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        rd = self._get_output_dir(trial)
        od = os.path.join(rd, cf)
        self._save_lora(od, model=model)
        if self.args.should_save:
            self.state.save_to_json(os.path.join(od, TRAINER_STATE_NAME))
        self._save_optimizer_and_scheduler(od)
        self._save_rng_state(od)


training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=1,  # 加权 dataset 已经膨胀到 ~250k+
    per_device_train_batch_size=2,  # MAX_LENGTH=1536 比较吃显存
    gradient_accumulation_steps=32,  # 等效 batch=64
    optim="paged_adamw_8bit",
    learning_rate=2e-5,  # 比 v3 低, 因为是 fine-tune
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    bf16=True,
    max_grad_norm=1.0,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
)

trainer = LoRATrainer(model=model, args=training_args, train_dataset=tokenized,
    data_collator=collator)

steps = len(tokenized) // (2 * 32)
print(f"\nBio-SFT v4:")
print(f"  Init from: SFT v3")
print(f"  Train rows (after weighting): {len(tokenized)}")
print(f"  MAX_LENGTH: {MAX_LENGTH}")
print(f"  Loss mask: prompt -100, answer only")
print(f"  Effective batch: 64, lr: 2e-5, total steps: ~{steps}")

trainer.train()

print(f"\nSaving to {OUTPUT_DIR}...", flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
torch.save(sd, os.path.join(OUTPUT_DIR, "lora_weights.pt"))
ew = model.get_input_embeddings().weight.detach().cpu()
torch.save(ew, os.path.join(OUTPUT_DIR, "embedding_weights.pt"))
tokenizer.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "bio_sft_v4_meta.json"), "w") as f:
    json.dump({
        "init_from": INIT_DIR,
        "train_file": TRAIN_FILE,
        "task_weights": TASK_WEIGHTS,
        "max_length": MAX_LENGTH,
        "loss_masking": "prompt=-100, answer-only",
        "prompt_template": "Alpaca pure (no chat tag)",
        "lr": 2e-5,
    }, f, indent=2)
print("Done!")
