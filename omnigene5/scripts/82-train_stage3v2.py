#!/usr/bin/env python
# coding: utf-8
"""
80b-train_stage3_v2.py

OmniGene-4-MM Stage 3 v2: properly continue from Stage 2 LoRA + stronger homology training.

Fixes from Stage 3 v1:
1. Load Stage 2 LoRA correctly (adapter name "stage2" to match saved keys)
2. Train 6000 steps (was 2000) -- ~1 epoch through the data
3. LR 5e-6 (was 2e-6) -- match Stage 1's effective LR
4. Mixed data: 50K text (homology x3 oversampled) + 10K vision (replay)

Goal:
- standard homology 90%+
- remote homology 75%+
- vision capability preserved (struct_recog 95+%, struct_cap 80+%)
- general text gen Cell/Mol > 50%
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import random
import torch
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
STAGE2_DIR  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2"
TEXT_FILE   = "/root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train_with_remote.jsonl"
VISION_JSONL = "/root/autodl-tmp/dnagpt/omnigene5/data/unified/train.jsonl"

OUTPUT_DIR  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v2"
CKPT_DIR    = "/root/autodl-fs/checkpoints_omnigene4_mm_stage3v2"
META_FILE   = f"{OUTPUT_DIR}/meta.json"

# Adapter name MUST match Stage 2's saved key
ADAPTER_NAME = "stage2"

LORA_R         = 64
LORA_ALPHA     = 128
LORA_DROPOUT   = 0.05
LR             = 5e-6
NUM_TRAIN_STEPS = 6000
PER_DEVICE_BATCH = 1
GRAD_ACCUM       = 32
MAX_LEN          = 1536
WARMUP_RATIO     = 0.03
SAVE_EVERY_N_STEPS = 1500

# Sampling
N_TEXT = 50000        # text instructions
N_VISION = 10000      # vision replay
HOMOLOGY_MULT = 3     # boost homology to ~45K x 3 = 135K -> 50K cap

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# =================== Setup ===================
print("Loading base + Stage 2 LoRA...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(STAGE2_DIR)
processor = AutoProcessor.from_pretrained(STAGE2_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print(f"  base loaded, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Freeze all
for p in model.parameters():
    p.requires_grad = False

# Inject LoRA with the SAME name as Stage 2 used
print(f"Injecting LoRA r={LORA_R}/alpha={LORA_ALPHA}, adapter name='{ADAPTER_NAME}'...", flush=True)
peft_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
inject_adapter_in_model(peft_cfg, model.model.language_model, adapter_name=ADAPTER_NAME)
model._hf_peft_config_loaded = True

# Load Stage 2 LoRA weights -- now keys should match
print("Loading Stage 2 LoRA into matching adapter...", flush=True)
ms = model.state_dict()
stage2_lora = torch.load(f"{STAGE2_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
missing = []
for k, v in stage2_lora.items():
    if k in ms:
        ms[k].copy_(v)
        loaded += 1
    else:
        missing.append(k)
print(f"  Stage 2 LoRA loaded: {loaded} tensors")
if missing:
    print(f"  WARNING: {len(missing)} keys not matched, sample: {missing[:3]}")
assert loaded > 400, f"Only {loaded} loaded - LoRA not properly continued!"

# Load Stage 2 embedding
stage2_emb = torch.load(f"{STAGE2_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(stage2_emb)
for p in model.get_input_embeddings().parameters():
    p.requires_grad = True
print(f"  Stage 2 embedding loaded ({stage2_emb.shape})")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  trainable: {trainable/1e9:.2f} B", flush=True)


# =================== Datasets ===================
print("\nLoading text corpus...", flush=True)
text_rows = []
with open(TEXT_FILE) as f:
    for line in f:
        text_rows.append(json.loads(line))
print(f"  text pool: {len(text_rows)}")


def is_homology(r):
    instr = r.get("instruction", "").lower()
    return ("homolog" in instr or "paraphrase" in instr or "structurally related" in instr)


# Oversample homology x3
homology_rows = [r for r in text_rows if is_homology(r)]
other_rows = [r for r in text_rows if not is_homology(r)]
print(f"  homology: {len(homology_rows)}, other: {len(other_rows)}")

# Build text pool: homology x3 + other (cap to N_TEXT)
expanded_text = homology_rows * HOMOLOGY_MULT + other_rows
random.shuffle(expanded_text)
text_subset = expanded_text[:N_TEXT]
print(f"  text subset: {len(text_subset)} ({sum(1 for r in text_subset if is_homology(r))} homology)")

print("\nLoading vision pool (Stage 1 same data)...", flush=True)
vision_rows = []
with open(VISION_JSONL) as f:
    for line in f:
        r = json.loads(line)
        if "vision" in r["modality"] and len(r.get("images", [])) == 1:
            vision_rows.append(r)
random.shuffle(vision_rows)
vision_subset = vision_rows[:N_VISION]
print(f"  vision subset: {len(vision_subset)}")


class MixedDataset(Dataset):
    def __init__(self, text_rows, vision_rows, tokenizer, processor, max_len=MAX_LEN):
        self.tok = tokenizer
        self.proc = processor
        self.max_len = max_len
        self.combined = []
        for r in text_rows: self.combined.append(("text", r))
        for r in vision_rows: self.combined.append(("vision", r))
        random.shuffle(self.combined)

    def __len__(self):
        return len(self.combined)

    def _vision_item(self, row):
        images = [Image.open(p).convert("RGB") for p in row["images"]]
        msgs = []
        for i, m in enumerate(row["messages"]):
            content = []
            if i == 0 and row["images"]:
                for _ in row["images"]:
                    content.append({"type": "image"})
            text_part = m["content"].replace("<image>\n", "").replace("<image>", "").strip()
            content.append({"type": "text", "text": text_part})
            msgs.append({"role": m["role"], "content": content})
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
        out = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
        for k in ("pixel_values", "image_position_ids", "mm_token_type_ids"):
            if k in inp:
                v = inp[k][0]
                if k == "mm_token_type_ids" and v.shape[0] > ids.shape[0]:
                    v = v[:ids.shape[0]]
                out[k] = v
        return out

    def _text_item(self, row):
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
            "mm_token_type_ids": torch.zeros_like(full),
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
                    print(f"[ds] {kind} {idx} failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
                idx = (idx + 1) % len(self.combined)
        return None


train_ds = MixedDataset(text_subset, vision_subset, tokenizer, processor)
print(f"  combined dataset: {len(train_ds)}")


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
        pix, ipos = [], []
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
        out = {"input_ids": ids, "labels": lbl, "attention_mask": am,
               "mm_token_type_ids": mm}
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
class S3v2Trainer(Trainer):
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
    save_total_limit=4,
    report_to="none",
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="no",
)

trainer = S3v2Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, data_collator=collator,
    processing_class=tokenizer,
)

print(f"\nStage 3 v2 (FIXED LoRA continuation):")
print(f"  Init: Stage 2 LoRA properly loaded (adapter='{ADAPTER_NAME}')")
print(f"  Data: {len(text_subset)} text (homology x{HOMOLOGY_MULT}) + {len(vision_subset)} vision replay")
print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
print(f"  LR: {LR}, max_steps: {NUM_TRAIN_STEPS}")
print(f"  Effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
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

with open(META_FILE, "w") as f:
    json.dump({
        "init_from_stage2": STAGE2_DIR,
        "stage": "stage3v2_homology_specialty_fixed",
        "adapter_name": ADAPTER_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lr": LR,
        "max_steps": NUM_TRAIN_STEPS,
        "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
        "homology_multiplier": HOMOLOGY_MULT,
        "n_text": len(text_subset),
        "n_vision": len(vision_subset),
    }, f, indent=2)
print("Done!")
