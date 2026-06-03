#!/usr/bin/env python
# coding: utf-8
"""
80-train_stage3.py

OmniGene-4-MM Stage 3: Homology task specialty training.

Goal: bring standard homology back from 59% to 90+%, remote from 57% to 75+%
without breaking vision or general text capability.

Init: Stage 2 LoRA r=64 + embedding (continue training)
Data: full OmniGene-4 v5 SFT corpus (199K) with 2x oversampling on
      homology-related samples (~45K -> 90K, total ~244K)
LR: 2e-6 (lower than Stage 2 to avoid disturbing vision)
Steps: 2000 (~13 GPU-h estimated)

Loss masking on prompt tokens (Alpaca standard).
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM,
    TrainingArguments, Trainer,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME
from peft import LoraConfig, inject_adapter_in_model

# =================== CONFIG ===================
BASE_MODEL  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
STAGE2_DIR  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2"
TRAIN_FILE  = "/root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train_with_remote.jsonl"

OUTPUT_DIR  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3"
CKPT_DIR    = "/root/autodl-fs/checkpoints_omnigene4_mm_stage3"
META_FILE   = f"{OUTPUT_DIR}/meta.json"

LORA_R         = 64
LORA_ALPHA     = 128
LORA_DROPOUT   = 0.05
LR             = 2e-6
NUM_TRAIN_STEPS = 2000
PER_DEVICE_BATCH = 1
GRAD_ACCUM       = 32
MAX_LEN          = 1536
WARMUP_RATIO     = 0.03
SAVE_EVERY_N_STEPS = 1000

# Homology oversampling
HOMOLOGY_MULT = 2

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# =================== Setup ===================
print("Loading base model OmniGene-4-v5-merged (BF16)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(STAGE2_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print(f"  loaded, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


# Freeze everything by default
for p in model.parameters():
    p.requires_grad = False

# Inject LoRA r=64 (same as Stage 2)
print(f"Injecting LoRA r={LORA_R}/alpha={LORA_ALPHA}...", flush=True)
peft_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
inject_adapter_in_model(peft_cfg, model.model.language_model, adapter_name="stage3")
model._hf_peft_config_loaded = True

# Load Stage 2 LoRA + embedding (CONTINUE from Stage 2)
print("Loading Stage 2 weights...", flush=True)
ms = model.state_dict()
stage2_lora = torch.load(f"{STAGE2_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
for k, v in stage2_lora.items():
    if k in ms:
        ms[k].copy_(v); loaded += 1
print(f"  Stage 2 LoRA loaded: {loaded} tensors")

stage2_emb = torch.load(f"{STAGE2_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(stage2_emb)
for p in model.get_input_embeddings().parameters():
    p.requires_grad = True
print(f"  Stage 2 embedding loaded ({stage2_emb.shape})")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  trainable: {trainable/1e9:.2f} B / {total/1e9:.2f} B", flush=True)


# =================== Dataset ===================
print("\nLoading SFT v5 corpus...", flush=True)
all_rows = []
with open(TRAIN_FILE) as f:
    for line in f:
        all_rows.append(json.loads(line))
print(f"  loaded {len(all_rows)} rows")


def is_homology(r):
    instr = r.get("instruction", "").lower()
    return ("homolog" in instr or "paraphrase" in instr or "structurally related" in instr)


# Oversample homology
homology_rows = [r for r in all_rows if is_homology(r)]
other_rows = [r for r in all_rows if not is_homology(r)]
print(f"  homology: {len(homology_rows)}, other: {len(other_rows)}")

# Build expanded corpus: homology x HOMOLOGY_MULT + other
combined = homology_rows * HOMOLOGY_MULT + other_rows
random.shuffle(combined)
print(f"  expanded total: {len(combined)} (homology mult x{HOMOLOGY_MULT})")


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=MAX_LEN):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        for attempt in range(5):
            row = self.rows[idx]
            try:
                instr = row["instruction"]
                inp = row.get("input", "") or ""
                out = row["output"]
                if inp.strip():
                    prompt = f"### Instruction:\n{instr}\n\n{inp}\n\n### Answer:\n"
                else:
                    prompt = f"### Instruction:\n{instr}\n\n### Answer:\n"
                ans = out + (self.tok.eos_token or "")
                p_ids = self.tok(prompt, add_special_tokens=False, truncation=True,
                                  max_length=self.max_len-256).input_ids
                a_ids = self.tok(ans, add_special_tokens=False, truncation=True,
                                  max_length=self.max_len-len(p_ids)).input_ids
                full = torch.tensor(p_ids + a_ids, dtype=torch.long)
                labels = torch.tensor([-100] * len(p_ids) + a_ids, dtype=torch.long)
                return {
                    "input_ids": full,
                    "labels": labels,
                    "attention_mask": torch.ones_like(full),
                    # Gemma4 needs mm_token_type_ids during training (all 0 for text)
                    "mm_token_type_ids": torch.zeros_like(full),
                }
            except Exception as e:
                if attempt == 0:
                    print(f"[ds] sample {idx} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
                idx = (idx + 1) % len(self.rows)
        return None


train_ds = TextDataset(combined, tokenizer)
print(f"  dataset size: {len(train_ds)}")


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        feats = [f for f in feats if f is not None]
        if not feats: return None
        ml = max(f["input_ids"].shape[0] for f in feats)
        ids = torch.full((len(feats), ml), self.pad_id, dtype=torch.long)
        lbl = torch.full((len(feats), ml), -100, dtype=torch.long)
        am = torch.zeros((len(feats), ml), dtype=torch.long)
        mm = torch.zeros((len(feats), ml), dtype=torch.long)
        for i, f in enumerate(feats):
            L = f["input_ids"].shape[0]
            ids[i, :L] = f["input_ids"]
            lbl[i, :L] = f["labels"]
            am[i, :L] = f["attention_mask"]
        return {"input_ids": ids, "labels": lbl, "attention_mask": am,
                "mm_token_type_ids": mm}


collator = Collator(tokenizer.pad_token_id or 0)


# =================== Trainer ===================
class S3Trainer(Trainer):
    def _save_lora(self, out_dir, model=None):
        m = model or self.model
        m_inner = m.module if hasattr(m, "module") else m
        os.makedirs(out_dir, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in m_inner.state_dict().items() if "lora_" in k}
        torch.save(sd, os.path.join(out_dir, "lora_weights.pt"))
        ew = m_inner.get_input_embeddings().weight.detach().cpu()
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
    output_dir=CKPT_DIR,
    max_steps=NUM_TRAIN_STEPS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="paged_adamw_8bit",
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    bf16=True,
    max_grad_norm=1.0,
    logging_steps=20,
    save_strategy="steps",
    save_steps=SAVE_EVERY_N_STEPS,
    save_total_limit=3,
    report_to="none",
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="no",
)

trainer = S3Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, data_collator=collator,
    processing_class=tokenizer,
)

print(f"\nOmniGene-4-MM Stage 3 (homology specialty):")
print(f"  Init from Stage 2 LoRA + embedding")
print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
print(f"  Train rows: {len(train_ds)} (homology x{HOMOLOGY_MULT})")
print(f"  Effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"  LR: {LR}, schedule: cosine + {WARMUP_RATIO*100:.0f}% warmup")
print(f"  Max steps: {NUM_TRAIN_STEPS}")
print()

trainer.train()


# =================== Final save ===================
print(f"\nSaving final to {OUTPUT_DIR}...", flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
torch.save(sd, os.path.join(OUTPUT_DIR, "lora_weights.pt"))
ew = model.get_input_embeddings().weight.detach().cpu()
torch.save(ew, os.path.join(OUTPUT_DIR, "embedding_weights.pt"))
tokenizer.save_pretrained(OUTPUT_DIR)

with open(META_FILE, "w") as f:
    json.dump({
        "init_from_stage2": STAGE2_DIR,
        "stage": "stage3_homology_specialty",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lr": LR,
        "max_steps": NUM_TRAIN_STEPS,
        "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
        "homology_multiplier": HOMOLOGY_MULT,
        "n_train_rows": len(train_ds),
    }, f, indent=2)
print("Done!")
