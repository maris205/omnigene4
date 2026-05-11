#!/usr/bin/env python
# coding: utf-8
"""
14-train_bio_sft_v2.py
Bio-SFT v2 训练

输入:
  - Base:   gemma-4-26B-A4B-it-bio
  - CPT:    outputs/gemma-4-26B-A4B-it-bio-cpt-v2 (0.6 epoch)
  - 数据:   omnigene_sft_v1_train.jsonl (179,576 条, {instruction,input,output})
输出:
  - outputs/OmniGene-4-v2-sft/

单卡 4-bit QLoRA, 自定义 LoRATrainer 支持 checkpoint 保存。
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Blackwell (sm_120) 不支持 torch._grouped_mm，强制走 loop fallback
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    logging,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME
from peft import LoraConfig, inject_adapter_in_model

# ================= 1. 配置 =================
BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
CPT_DIR = "/root/autodl-tmp/dnagpt/outputs/gemma-4-26B-A4B-it-bio-cpt-v2"
TRAIN_FILE = "/root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v2-sft"
CHECKPOINT_DIR = "/root/autodl-tmp/dnagpt/checkpoints_sft_v2"

MAX_LENGTH = 1024

# ================= 2. 加载数据 =================
print("Loading Bio-SFT v2 dataset...", flush=True)
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"  Total records: {len(dataset)}")

# ================= 3. 格式化函数 =================
def format_instruction(instruction, input_text, output):
    """Alpaca 风格 + 保留原 prompt 结构的一部分"""
    if input_text and input_text.strip():
        prompt = (
            f"<User>\n### Instruction:\n{instruction}\n\n"
            f"{input_text}\n"
            f"### Answer:\n<Assistant>\n{output}"
        )
    else:
        prompt = (
            f"<User>\n### Instruction:\n{instruction}\n\n"
            f"### Answer:\n<Assistant>\n{output}"
        )
    return prompt

# ================= 4. 加载模型 (4-bit) =================
print("Loading model (4-bit)...", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

tokenizer = AutoTokenizer.from_pretrained(CPT_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 5. 注入 LoRA =================
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
for param in model.parameters():
    param.requires_grad = False
inject_adapter_in_model(peft_config, model.model.language_model, adapter_name="default")
model._hf_peft_config_loaded = True

# ================= 6. 加载 CPT 权重 =================
print("Loading CPT LoRA...", flush=True)
cpt_lora = torch.load(f"{CPT_DIR}/lora_weights.pt", map_location="cpu")
model_state = model.state_dict()
loaded_lora = 0
for k, v in cpt_lora.items():
    if k in model_state:
        model_state[k].copy_(v)
        loaded_lora += 1
print(f"  Loaded {loaded_lora} CPT LoRA tensors.")

print("Loading CPT embedding...", flush=True)
embed_weights = torch.load(f"{CPT_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(embed_weights)

# 解冻 embedding 继续训练 (保持 bfloat16)
for param in model.get_input_embeddings().parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")

# ================= 7. Tokenize =================
def tokenize_function(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        prompt = format_instruction(
            examples["instruction"][i],
            examples["input"][i],
            examples["output"][i],
        ) + tokenizer.eos_token
        texts.append(prompt)
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    return tokenized

print("Tokenizing dataset...", flush=True)
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=8,
)

# ================= 8. Collator =================
class BioSFTCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, features):
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids = f["input_ids"][:max_len]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            labels.append(ids + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(len(features), max_len, dtype=torch.long),
        }

collator = BioSFTCollator(tokenizer, MAX_LENGTH)

# ================= 9. 自定义 Trainer =================
class LoRATrainer(Trainer):
    def _save_lora_checkpoint(self, output_dir, model=None):
        model = model or self.model
        base_model = model.module if hasattr(model, "module") else model
        os.makedirs(output_dir, exist_ok=True)
        lora_state_dict = {k: v.detach().cpu() for k, v in base_model.state_dict().items() if "lora_" in k}
        torch.save(lora_state_dict, os.path.join(output_dir, "lora_weights.pt"))
        embed_weight = base_model.get_input_embeddings().weight.detach().cpu()
        torch.save(embed_weight, os.path.join(output_dir, "embedding_weights.pt"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        self._save_lora_checkpoint(output_dir)

    def _save_checkpoint(self, model, trial):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_lora_checkpoint(output_dir, model=model)
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        self._save_optimizer_and_scheduler(output_dir)
        self._save_rng_state(output_dir)

# ================= 10. 训练 =================
print("Setting up training...", flush=True)
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,         # 等效 batch = 4 × 1 × 16 = 64
    optim="paged_adamw_8bit",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    bf16=True,
    max_grad_norm=1.0,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
)

trainer = LoRATrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator,
)

total_steps = len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
print(f"\nStarting Bio-SFT v2 training...")
print(f"  Epochs: 1")
print(f"  Batch: {training_args.per_device_train_batch_size} × 1 GPU × {training_args.gradient_accumulation_steps} accum = 64")
print(f"  Total steps: ~{total_steps}")

trainer.train()

# ================= 11. 保存最终 =================
print(f"\nSaving Bio-SFT v2 to {OUTPUT_DIR}...", flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
lora_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
torch.save(lora_state_dict, os.path.join(OUTPUT_DIR, "lora_weights.pt"))
embed_weight = model.get_input_embeddings().weight.detach().cpu()
torch.save(embed_weight, os.path.join(OUTPUT_DIR, "embedding_weights.pt"))
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "bio_sft_v2_meta.json"), "w") as f:
    json.dump({
        "base_model": BASE_MODEL,
        "cpt_dir": CPT_DIR,
        "train_file": TRAIN_FILE,
        "train_rows": len(dataset),
        "trainable_params": trainable,
        "lora_r": 64,
        "lora_alpha": 128,
        "max_length": MAX_LENGTH,
    }, f, indent=2)

print("Done! OmniGene-4 v2 SFT saved.")
