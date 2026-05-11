#!/usr/bin/env python
# coding: utf-8
"""
17-train_bio_sft_v3_remote.py
Bio-SFT v3 (Remote-augmented)

输入:
  - Base:    gemma-4-26B-A4B-it-bio
  - SFT v2:  outputs/OmniGene-4-v2-sft (Standard 100%, Remote 56.55%)
  - 数据:    omnigene_sft_v1_train_with_remote.jsonl (199,576 = 179.5k base + 20k remote)
输出:
  - outputs/OmniGene-4-v3-sft-remote/

策略: 从 v2 SFT 继续训, 学习远缘同源的"序列相似度低但同源"模式。
单卡 4-bit QLoRA, lr=2e-5 (比 v2 5e-5 小, 避免破坏已有能力), 1 epoch.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME
from peft import LoraConfig, inject_adapter_in_model

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_V2_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v2-sft"
TRAIN_FILE = "/root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train_with_remote.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v3-sft-remote"
CHECKPOINT_DIR = "/root/autodl-tmp/dnagpt/checkpoints_sft_v3"
MAX_LENGTH = 1024

print("Loading dataset...", flush=True)
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"  rows: {len(dataset)}")


def format_instruction(instruction, input_text, output):
    if input_text and input_text.strip():
        return (
            f"<User>\n### Instruction:\n{instruction}\n\n"
            f"{input_text}\n"
            f"### Answer:\n<Assistant>\n{output}"
        )
    return (
        f"<User>\n### Instruction:\n{instruction}\n\n"
        f"### Answer:\n<Assistant>\n{output}"
    )


print("Loading model (4-bit)...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map={"":0})
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

tokenizer = AutoTokenizer.from_pretrained(SFT_V2_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','router.proj'])
for p in model.parameters():
    p.requires_grad = False
inject_adapter_in_model(peft_config, model.model.language_model, adapter_name="default")
model._hf_peft_config_loaded = True

print("Loading SFT v2 LoRA...", flush=True)
sft_lora = torch.load(f"{SFT_V2_DIR}/lora_weights.pt", map_location="cpu")
ms = model.state_dict()
n = 0
for k, v in sft_lora.items():
    if k in ms:
        ms[k].copy_(v); n += 1
print(f"  loaded {n} tensors")

print("Loading SFT v2 embedding...", flush=True)
emb = torch.load(f"{SFT_V2_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(emb)

for p in model.get_input_embeddings().parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")


def tok_fn(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        s = format_instruction(examples["instruction"][i], examples["input"][i],
                               examples["output"][i]) + tokenizer.eos_token
        texts.append(s)
    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)

print("Tokenizing...", flush=True)
tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names, num_proc=8)


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
            pl = ml - len(x)
            ids.append(x + [self.pad]*pl)
            lbl.append(x + [-100]*pl)
            am.append([1]*len(x) + [0]*pl)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lbl, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(len(feats), ml, dtype=torch.long),
        }

collator = Collator(tokenizer, MAX_LENGTH)


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
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    optim="paged_adamw_8bit",
    learning_rate=2e-5,
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

steps = len(tokenized) // (4 * 16)
print(f"\nBio-SFT v3 (Remote-augmented):")
print(f"  Init from: SFT v2 (Standard 100%, Remote 56.55%)")
print(f"  Train rows: {len(tokenized)}")
print(f"  Effective batch: 64, lr: 2e-5, total steps: ~{steps}")

trainer.train()

print(f"\nSaving to {OUTPUT_DIR}...", flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
torch.save(sd, os.path.join(OUTPUT_DIR, "lora_weights.pt"))
ew = model.get_input_embeddings().weight.detach().cpu()
torch.save(ew, os.path.join(OUTPUT_DIR, "embedding_weights.pt"))
tokenizer.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "bio_sft_v3_meta.json"), "w") as f:
    json.dump({
        "init_from": SFT_V2_DIR,
        "train_file": TRAIN_FILE,
        "train_rows": len(dataset),
        "remote_added": 20000,
        "lr": 2e-5,
        "epochs": 1,
        "max_length": MAX_LENGTH,
    }, f, indent=2)
print("Done!")
