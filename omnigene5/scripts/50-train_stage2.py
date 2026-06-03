#!/usr/bin/env python
# coding: utf-8
"""
50-train_stage2.py

OmniGene-4-MM Stage 2: Recover text reasoning while keeping vision adaptation.

Init from: OmniGene-4-MM-stage1 LoRA + embedding (NOT v5-merged).
Data mix:
  - 50% vision-bearing samples (same 95K from Stage 1)
  - 50% OmniGene-4 SFT v5 text-only corpus (199K Alpaca instructions)

LoRA: r=64 / alpha=128 (UPGRADED from r=32 to give both capabilities room).
LR: 3e-6 cosine + 3% warmup (lower than Stage 1 to avoid disturbing vision learning).
Steps: 3000 (~25 GPU-h estimated at 30 sec/step).

Loss masking on prompt tokens (only assistant turn contributes gradient).
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
STAGE1_DIR  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage1"
VISION_JSONL = "/root/autodl-tmp/dnagpt/omnigene5/data/unified/train.jsonl"
TEXT_JSONL  = "/root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train_with_remote.jsonl"

OUTPUT_DIR  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2"
CKPT_DIR    = "/root/autodl-fs/checkpoints_omnigene4_mm_stage2"
META_FILE   = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2/meta.json"

LORA_R         = 64                # upgraded from Stage 1 (r=32)
LORA_ALPHA     = 128
LORA_DROPOUT   = 0.05
LR             = 3e-6              # lower than Stage 1
NUM_TRAIN_STEPS = 3000
PER_DEVICE_BATCH = 1
GRAD_ACCUM       = 32
MAX_LEN          = 2048
WARMUP_RATIO     = 0.03
SAVE_EVERY_N_STEPS = 1000

# Sampling
N_VISION = 50000          # subset of vision pool
N_TEXT   = 50000          # subset of text pool
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# =================== Setup ===================
print("Loading base model OmniGene-4-v5-merged (BF16)...", flush=True)
processor = AutoProcessor.from_pretrained(STAGE1_DIR)
tokenizer = AutoTokenizer.from_pretrained(STAGE1_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print(f"  loaded, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


# Freeze everything by default
for p in model.parameters():
    p.requires_grad = False

# Inject LoRA r=64 (NEW, larger than Stage 1)
print(f"Injecting LoRA r={LORA_R}/alpha={LORA_ALPHA}...", flush=True)
peft_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
inject_adapter_in_model(peft_cfg, model.model.language_model, adapter_name="stage2")
model._hf_peft_config_loaded = True

# Load Stage 1 embedding (vision-aware)
stage1_emb = torch.load(f"{STAGE1_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(stage1_emb)
print(f"  Stage 1 embedding loaded ({stage1_emb.shape})", flush=True)

# Note: Stage 1 LoRA was r=32 — we cannot directly load it into r=64 LoRA layers.
# Instead, we use Stage 1 embedding as warm-start; LoRA starts fresh at r=64.
# The base model has all v5 knowledge baked in, so we recover quickly.

# Unfreeze embedding (vision tokens flow through embed)
for p in model.get_input_embeddings().parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  trainable: {trainable/1e9:.2f} B / {total/1e9:.2f} B ({100*trainable/total:.2f}%)", flush=True)


# =================== Datasets ===================
print("Loading datasets...", flush=True)

# Vision pool (single-image only, paths verified)
vision_rows = []
with open(VISION_JSONL) as f:
    for line in f:
        r = json.loads(line)
        if "vision" in r["modality"] and len(r.get("images", [])) == 1:
            vision_rows.append(r)
print(f"  vision pool: {len(vision_rows)}", flush=True)

# Text pool (Alpaca instruction format)
text_rows = []
with open(TEXT_JSONL) as f:
    for line in f:
        r = json.loads(line)
        text_rows.append(r)
print(f"  text pool: {len(text_rows)}", flush=True)

# Subsample
random.shuffle(vision_rows)
random.shuffle(text_rows)
vision_rows = vision_rows[:N_VISION]
text_rows = text_rows[:N_TEXT]
print(f"  vision sampled: {len(vision_rows)}, text sampled: {len(text_rows)}", flush=True)


class MixedDataset(Dataset):
    """Mixes vision-text and text-only samples."""
    def __init__(self, vision_rows, text_rows, processor, tokenizer, max_len=MAX_LEN):
        self.vision_rows = vision_rows
        self.text_rows = text_rows
        self.proc = processor
        self.tok = tokenizer
        self.max_len = max_len
        # Interleave: alternate between vision and text
        self.combined = []
        for i in range(max(len(vision_rows), len(text_rows))):
            if i < len(vision_rows):
                self.combined.append(("vision", vision_rows[i]))
            if i < len(text_rows):
                self.combined.append(("text", text_rows[i]))
        random.shuffle(self.combined)

    def __len__(self):
        return len(self.combined)

    def _build_chat(self, row):
        msgs = []
        for i, m in enumerate(row["messages"]):
            content = []
            if i == 0 and row["images"]:
                for _ in row["images"]:
                    content.append({"type": "image"})
            text_part = m["content"].replace("<image>\n", "").replace("<image>", "").strip()
            content.append({"type": "text", "text": text_part})
            msgs.append({"role": m["role"], "content": content})
        return msgs

    def _vision_item(self, row):
        images = [Image.open(p).convert("RGB") for p in row["images"]]
        msgs = self._build_chat(row)
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        inp = self.proc(text=text, images=images, return_tensors="pt")
        ids = inp["input_ids"][0]
        labels = ids.clone()
        if msgs[-1]["role"] == "assistant":
            prefix_msgs = msgs[:-1]
            prefix_text = self.proc.apply_chat_template(prefix_msgs, add_generation_prompt=True, tokenize=False)
            prefix_inp = self.proc(text=prefix_text, images=images, return_tensors="pt")
            prompt_len = prefix_inp["input_ids"].shape[1]
            if prompt_len < ids.shape[0]:
                labels[:prompt_len] = -100
        if ids.shape[0] > self.max_len:
            ids = ids[:self.max_len]
            labels = labels[:self.max_len]
        out = {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones_like(ids),
        }
        for k in ("pixel_values", "image_position_ids", "mm_token_type_ids"):
            if k in inp:
                v = inp[k][0]
                if k == "mm_token_type_ids" and v.shape[0] > ids.shape[0]:
                    v = v[:ids.shape[0]]
                out[k] = v
        return out

    def _text_item(self, row):
        # Alpaca format: instruction / input / output
        instr = row["instruction"]
        inp = row.get("input", "") or ""
        out = row["output"]
        if inp.strip():
            prompt = f"### Instruction:\n{instr}\n\n{inp}\n\n### Answer:\n"
        else:
            prompt = f"### Instruction:\n{instr}\n\n### Answer:\n"
        ans_with_eos = out + (self.tok.eos_token or "")
        prompt_ids = self.tok(prompt, add_special_tokens=False, truncation=True,
                              max_length=self.max_len - 256).input_ids
        ans_ids = self.tok(ans_with_eos, add_special_tokens=False, truncation=True,
                           max_length=self.max_len - len(prompt_ids)).input_ids
        full_ids = torch.tensor(prompt_ids + ans_ids, dtype=torch.long)
        labels = torch.tensor([-100] * len(prompt_ids) + ans_ids, dtype=torch.long)
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
            # Gemma4 requires mm_token_type_ids during training; all zeros = pure text
            "mm_token_type_ids": torch.zeros_like(full_ids),
        }

    def __getitem__(self, idx):
        for attempt in range(5):
            kind, row = self.combined[idx]
            try:
                if kind == "vision":
                    return self._vision_item(row)
                else:
                    return self._text_item(row)
            except Exception as e:
                if attempt == 0:
                    print(f"[ds] sample {idx} ({kind}) failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
                idx = (idx + 1) % len(self.combined)
        return None


train_ds = MixedDataset(vision_rows, text_rows, processor, tokenizer)
print(f"  combined dataset size: {len(train_ds)}", flush=True)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        feats = [f for f in feats if f is not None]
        if not feats:
            return None
        ml = max(f["input_ids"].shape[0] for f in feats)
        ids = torch.full((len(feats), ml), self.pad_id, dtype=torch.long)
        lbl = torch.full((len(feats), ml), -100, dtype=torch.long)
        am = torch.zeros((len(feats), ml), dtype=torch.long)
        mm = torch.zeros((len(feats), ml), dtype=torch.long)
        pix, ipos = [], []
        has_pix_any = any("pixel_values" in f for f in feats)
        has_mm_any = any("mm_token_type_ids" in f for f in feats)
        for i, f in enumerate(feats):
            L = f["input_ids"].shape[0]
            ids[i, :L] = f["input_ids"]
            lbl[i, :L] = f["labels"]
            am[i, :L] = f["attention_mask"]
            if "mm_token_type_ids" in f:
                mm_len = min(f["mm_token_type_ids"].shape[0], L)
                mm[i, :mm_len] = f["mm_token_type_ids"][:mm_len]
            if "pixel_values" in f:
                pix.append(f["pixel_values"])
                if "image_position_ids" in f:
                    ipos.append(f["image_position_ids"])
        out = {"input_ids": ids, "labels": lbl, "attention_mask": am}
        if has_mm_any:
            out["mm_token_type_ids"] = mm
        if pix:
            try:
                out["pixel_values"] = torch.stack(pix)
                if ipos: out["image_position_ids"] = torch.stack(ipos)
            except Exception:
                out["pixel_values"] = pix
                if ipos: out["image_position_ids"] = ipos
        return out


collator = Collator(tokenizer.pad_token_id or 0)


# =================== Trainer ===================
class S2Trainer(Trainer):
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

trainer = S2Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, data_collator=collator,
    processing_class=tokenizer,
)

print(f"\nOmniGene-4-MM Stage 2:")
print(f"  Init from: Stage 1 embedding (Stage 1 r=32 LoRA discarded; new r={LORA_R} LoRA)")
print(f"  Data mix: {len(vision_rows)} vision + {len(text_rows)} text = {len(train_ds)} total")
print(f"  Effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"  LR: {LR}, schedule: cosine + {WARMUP_RATIO*100:.0f}% warmup")
print(f"  Max steps: {NUM_TRAIN_STEPS}")
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
        "init_from_stage1": STAGE1_DIR,
        "stage": "stage2_text_recovery",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lr": LR,
        "max_steps": NUM_TRAIN_STEPS,
        "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
        "n_vision": len(vision_rows),
        "n_text": len(text_rows),
        "modalities": ["vision", "text", "sequence"],
        "vision_tower_frozen": True,
        "note": "Stage 1 r=32 LoRA discarded; fresh r=64 LoRA on top of Stage 1 embedding + v5 base",
    }, f, indent=2)
print("Done!")
