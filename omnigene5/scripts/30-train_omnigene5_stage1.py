#!/usr/bin/env python
# coding: utf-8
"""
30-train_omnigene5_stage1.py

OmniGene-5 Stage 1: Vision-LM warmup.

Init from: OmniGene-4-v5-merged (49 GB BF16, contains CPT+SFT v2-v5+dual-head)
Frozen:    vision_tower (visual encoder, 27-layer ViT)
Trained:   embedding + LoRA on language_model MoE layers (q/k/v/o, FFN, router.proj)

Goal: teach the model that "an image arriving in the input embedding stream
is just another modality" -- the language MoE learns to route vision-derived
tokens through appropriate experts.

Loss: causal LM with prompt-mask (loss on assistant turn only, like SFT v4).

Data: vision-bearing subset of unified/train.jsonl (Vis-CheBI20 + PubMedVision
+ biomed_vis + ChartQA + HPA10M).
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import random
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM,
    TrainingArguments, Trainer,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME
from peft import LoraConfig, inject_adapter_in_model

# =================== CONFIG ===================
BASE_MODEL  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
DATA_JSONL  = "/root/autodl-tmp/dnagpt/omnigene5/data/unified/train.jsonl"
VAL_JSONL   = "/root/autodl-tmp/dnagpt/omnigene5/data/unified/val.jsonl"
OUTPUT_DIR  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-5-stage1"
CKPT_DIR    = "/root/autodl-fs/checkpoints_omnigene5_stage1"
META_FILE   = "/root/autodl-tmp/dnagpt/outputs/OmniGene-5-stage1/meta.json"

LORA_R         = 32
LORA_ALPHA     = 64
LORA_DROPOUT   = 0.05
LR             = 5e-6
NUM_TRAIN_STEPS = 2000           # ~30 GPU-hours est.
PER_DEVICE_BATCH = 1             # vision adds tokens, keep batch small
GRAD_ACCUM       = 32            # effective batch 32
MAX_LEN          = 2048          # vision adds 2520 tokens; need bigger ctx
WARMUP_RATIO     = 0.03
SAVE_EVERY_N_STEPS = 1000

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# =================== Setup ===================
print("Loading base model OmniGene-4-v5-merged (BF16)...", flush=True)
processor = AutoProcessor.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print(f"  loaded, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


# Freeze everything by default
for p in model.parameters():
    p.requires_grad = False

# Inject LoRA on language model only (vision tower stays frozen)
print("Injecting LoRA on language_model...", flush=True)
peft_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
inject_adapter_in_model(peft_cfg, model.model.language_model, adapter_name="omnigene5")
model._hf_peft_config_loaded = True

# Unfreeze embedding (vision tokens flow through embed_vision -> language model)
for p in model.get_input_embeddings().parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  trainable: {trainable/1e9:.2f} B / {total/1e9:.2f} B "
      f"({100*trainable/total:.2f}%)", flush=True)


# =================== Dataset ===================
print("Loading dataset...", flush=True)


def load_records(path, vision_only=True, single_image_only=True):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if vision_only and "vision" not in r["modality"]:
                continue
            if single_image_only:
                # Stage 1: only single-image samples for clean alignment
                if len(r.get("images", [])) != 1:
                    continue
            rows.append(r)
    return rows


# Stage 1: vision-bearing samples only. ~243K available; use shuffled subset.
train_rows = load_records(DATA_JSONL, vision_only=True)
val_rows = load_records(VAL_JSONL, vision_only=True)
random.shuffle(train_rows)
print(f"  train rows (vision): {len(train_rows)}, val rows: {len(val_rows)}", flush=True)


class MultiModalDataset(Dataset):
    def __init__(self, rows, processor, max_len=MAX_LEN):
        self.rows = rows
        self.proc = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def _build_chat(self, row):
        msgs = []
        for i, m in enumerate(row["messages"]):
            content = []
            if i == 0 and row["images"]:
                # First user turn carries the image
                for _ in row["images"]:
                    content.append({"type": "image"})
            text_part = m["content"]
            # Strip the "<image>" placeholder if it appears in text (we use struct format)
            text_part = text_part.replace("<image>\n", "").replace("<image>", "").strip()
            content.append({"type": "text", "text": text_part})
            msgs.append({"role": m["role"], "content": content})
        return msgs

    def __getitem__(self, idx):
        for attempt in range(5):
            row = self.rows[idx]
            try:
                images = [Image.open(p).convert("RGB") for p in row["images"]]
                msgs = self._build_chat(row)
                text = self.proc.apply_chat_template(
                    msgs, add_generation_prompt=False, tokenize=False
                )
                inp = self.proc(text=text, images=images, return_tensors="pt")
                # Build labels with prompt masking
                ids = inp["input_ids"][0]
                labels = ids.clone()
                if msgs[-1]["role"] == "assistant":
                    prefix_msgs = msgs[:-1]
                    prefix_text = self.proc.apply_chat_template(
                        prefix_msgs, add_generation_prompt=True, tokenize=False
                    )
                    prefix_inp = self.proc(text=prefix_text, images=images, return_tensors="pt")
                    prompt_len = prefix_inp["input_ids"].shape[1]
                    if prompt_len < ids.shape[0]:
                        labels[:prompt_len] = -100
                # Truncate
                if ids.shape[0] > self.max_len:
                    ids = ids[:self.max_len]
                    labels = labels[:self.max_len]
                am = torch.ones_like(ids)
                out = {
                    "input_ids": ids,
                    "labels": labels,
                    "attention_mask": am,
                }
                # Pass-through ALL vision-related fields if present
                for k in ("pixel_values", "image_position_ids", "mm_token_type_ids"):
                    if k in inp:
                        out[k] = inp[k][0]
                # mm_token_type_ids has same length as input_ids - truncate alignment
                if "mm_token_type_ids" in out and out["mm_token_type_ids"].shape[0] > ids.shape[0]:
                    out["mm_token_type_ids"] = out["mm_token_type_ids"][:ids.shape[0]]
                return out
            except Exception as e:
                if attempt == 0:
                    print(f"[ds] sample {idx} src={row.get('source','?')} failed: {type(e).__name__}: {e}", flush=True)
                idx = (idx + 1) % len(self.rows)
        return None  # gives up; collator filters None


train_ds = MultiModalDataset(train_rows, processor)
val_ds = MultiModalDataset(val_rows, processor)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        feats = [f for f in feats if f is not None]
        if not feats:
            return None
        ml = max(f["input_ids"].shape[0] for f in feats)
        ids = torch.zeros((len(feats), ml), dtype=torch.long).fill_(self.pad_id)
        lbl = torch.full((len(feats), ml), -100, dtype=torch.long)
        am = torch.zeros((len(feats), ml), dtype=torch.long)
        # mm_token_type_ids: 0 = text, 1 = vision (Gemma4 convention)
        mm = torch.zeros((len(feats), ml), dtype=torch.long)
        pix = []
        ipos = []
        has_pix = "pixel_values" in feats[0]
        has_mm = "mm_token_type_ids" in feats[0]
        for i, f in enumerate(feats):
            L = f["input_ids"].shape[0]
            ids[i, :L] = f["input_ids"]
            lbl[i, :L] = f["labels"]
            am[i, :L] = f["attention_mask"]
            if has_mm:
                mm_len = min(f["mm_token_type_ids"].shape[0], L)
                mm[i, :mm_len] = f["mm_token_type_ids"][:mm_len]
            if has_pix:
                pix.append(f["pixel_values"])
                if "image_position_ids" in f:
                    ipos.append(f["image_position_ids"])
        out = {
            "input_ids": ids,
            "labels": lbl,
            "attention_mask": am,
        }
        if has_mm:
            out["mm_token_type_ids"] = mm
        if has_pix:
            try:
                out["pixel_values"] = torch.stack(pix)
                if ipos:
                    out["image_position_ids"] = torch.stack(ipos)
            except Exception:
                out["pixel_values"] = pix
                if ipos:
                    out["image_position_ids"] = ipos
        return out


collator = Collator(tokenizer.pad_token_id or 0)


# =================== Custom trainer with LoRA-only checkpointing ===================
class OG5Trainer(Trainer):
    def _save_lora(self, out_dir, model=None):
        m = model or self.model
        m_inner = m.module if hasattr(m, "module") else m
        os.makedirs(out_dir, exist_ok=True)
        # save LoRA weights only
        sd = {k: v.detach().cpu() for k, v in m_inner.state_dict().items() if "lora_" in k}
        torch.save(sd, os.path.join(out_dir, "lora_weights.pt"))
        # save embedding (we changed it)
        ew = m_inner.get_input_embeddings().weight.detach().cpu()
        torch.save(ew, os.path.join(out_dir, "embedding_weights.pt"))
        # save tokenizer + processor
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
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="no",   # quick path; evaluate offline post-hoc
)

trainer = OG5Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, data_collator=collator,
    processing_class=tokenizer,
)

print(f"\nOmniGene-5 Stage 1:")
print(f"  Init from: {BASE_MODEL}")
print(f"  Vision tower: FROZEN")
print(f"  LoRA target: language_model q/k/v/o, gate/up/down, router.proj (r={LORA_R}, alpha={LORA_ALPHA})")
print(f"  Embedding: trainable")
print(f"  Train rows (vision-bearing): {len(train_rows):,}")
print(f"  Effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"  LR: {LR}, schedule: cosine + {WARMUP_RATIO*100:.0f}% warmup")
print(f"  Max steps: {NUM_TRAIN_STEPS:,}")
print(f"  Checkpoints: {CKPT_DIR}/checkpoint-{{step}}")
print(f"  Final output: {OUTPUT_DIR}")
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
processor.save_pretrained(OUTPUT_DIR)

os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
with open(META_FILE, "w") as f:
    json.dump({
        "init_from": BASE_MODEL,
        "stage": "stage1_vision_warmup",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lr": LR,
        "max_steps": NUM_TRAIN_STEPS,
        "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
        "train_rows": len(train_rows),
        "modalities": ["vision", "text"],
        "vision_tower_frozen": True,
    }, f, indent=2)
print("Done!")
