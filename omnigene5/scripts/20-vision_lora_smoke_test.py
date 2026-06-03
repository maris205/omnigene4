#!/usr/bin/env python
# coding: utf-8
"""
20-vision_lora_smoke_test.py

Smoke test for OmniGene-5 vision-LoRA pipeline. Goal: verify in ~30 min that:

1. OmniGene-4-v5-merged (BF16, contains CPT + SFT v2-v5 + dual-head) loads OK
   on RTX Pro 6000 96GB
2. Vision tokens get routed through MoE correctly (one image -> N tokens)
3. LoRA can be attached to vision tower + language MoE layers
4. A 5-sample mini-batch trains for 10 steps with loss decreasing

Output: console log + /tmp/omnigene5_smoke.json with metrics.

If this passes, full OmniGene-5 training can begin.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, inject_adapter_in_model

# Use the merged v5 model (49GB BF16, includes CPT + SFT v2-v5 + dual-head)
# This is the natural starting point for OmniGene-5 (cumulative training).
BASE_MODEL = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
DATA_JSONL = "/root/autodl-tmp/dnagpt/omnigene5/data/unified/train.jsonl"
OUT_REPORT = "/tmp/omnigene5_smoke.json"

print("=" * 60)
print("OmniGene-5 Vision-LoRA Smoke Test (BF16, no quantization)")
print(f"Base: {BASE_MODEL}")
print("=" * 60)


def info(msg):
    print(f"[smoke] {msg}", flush=True)


# ============== Step 1: load model + processor ==============
info("Step 1/5: loading BF16 model (~49 GB) + AutoProcessor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
torch.cuda.empty_cache()
mem_used = torch.cuda.memory_allocated() / 1e9
info(f"  loaded: {model.config.architectures[0]}")
info(f"  GPU memory used: {mem_used:.1f} GB")
info(f"  loaded: {model.config.architectures[0]}")
info(f"  vision config: hidden={model.config.vision_config.hidden_size}, layers={model.config.vision_config.num_hidden_layers}")
info(f"  text config: hidden={model.config.text_config.hidden_size}, layers={model.config.text_config.num_hidden_layers}")


# ============== Step 2: explore module hierarchy ==============
info("\nStep 2/5: module hierarchy probe...")
have_vision = hasattr(model.model, "vision_tower") or hasattr(model, "vision_tower")
have_lm = hasattr(model.model, "language_model")
info(f"  vision_tower attr: {have_vision}")
info(f"  language_model attr: {have_lm}")

# Find target modules in vision tower
vision_targets = []
for name, mod in model.named_modules():
    if "vision_tower" in name and isinstance(mod, nn.Linear):
        # Pick q/k/v projections in attention only
        if any(k in name.split(".")[-1] for k in ["q_proj", "k_proj", "v_proj"]):
            vision_targets.append(name)
info(f"  vision linear modules with q/k/v: {len(vision_targets)}")
if vision_targets:
    info(f"    sample: {vision_targets[:3]}")
    info(f"    last 4 layers' modules: {vision_targets[-12:]}")


# ============== Step 3: attach a fresh LoRA on the merged v5 model ==============
info("\nStep 3/5: attach LoRA on merged v5 (no v5 weight reload, already in BF16)...")
# Since BASE_MODEL is OmniGene-4-v5-merged, all CPT + SFT v2-v5 weights are already
# baked in. We only need a *new* LoRA layer for OmniGene-5 vision adaptation.
new_lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
inject_adapter_in_model(new_lora_cfg, model.model.language_model, adapter_name="omnigene5")

# Make sure embedding stays trainable (vision tokens need to flow through it)
for p in model.get_input_embeddings().parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
info(f"  fresh LoRA injected (r=32, alpha=64), trainable params: {trainable:,}")


# ============== Step 4: forward pass with image input ==============
info("\nStep 4/5: forward pass with one image-text sample...")
# Find one Vis-CheBI20 sample with image
samples = []
with open(DATA_JSONL) as f:
    for line in f:
        r = json.loads(line)
        if r["images"] and "vis_chebi20" in r["source"]:
            if all(os.path.exists(p) for p in r["images"]):
                samples.append(r)
                if len(samples) >= 5: break

info(f"  found {len(samples)} Vis-CheBI20 samples with valid images")
if not samples:
    info("  no valid sample! exit")
    exit(1)

# Build a chat prompt + image
sample = samples[0]
images = [Image.open(p).convert("RGB") for p in sample["images"]]
messages = sample["messages"]
# Use processor to build inputs
text = processor.apply_chat_template(
    [{"role": m["role"], "content": [{"type": "image"}] if i == 0 else []
                                      + [{"type": "text", "text": m["content"]}]}
     for i, m in enumerate(messages)],
    add_generation_prompt=False, tokenize=False,
)
info(f"  prompt template (first 200 chars): {text[:200]}")

inputs = processor(text=text, images=images, return_tensors="pt").to(model.device)
info(f"  input_ids shape: {inputs.input_ids.shape}")
info(f"  pixel_values shape: {inputs.pixel_values.shape if 'pixel_values' in inputs else 'NONE'}")

with torch.no_grad():
    out = model(**inputs)
info(f"  forward pass OK: logits shape {out.logits.shape}")
del out


# ============== Step 5: 10-step mini-batch training ==============
info("\nStep 5/5: 10-step training on 5 samples (no vision LoRA in v1 -- verify text path first)...")
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5,
)

losses = []
for step in range(10):
    s = samples[step % len(samples)]
    images = [Image.open(p).convert("RGB") for p in s["images"]]
    messages = s["messages"]
    text = processor.apply_chat_template(
        [{"role": m["role"], "content": [{"type": "image"}] if i == 0 else []
                                          + [{"type": "text", "text": m["content"]}]}
         for i, m in enumerate(messages)],
        add_generation_prompt=False, tokenize=False,
    )
    inp = processor(text=text, images=images, return_tensors="pt").to(model.device)
    inp["labels"] = inp.input_ids.clone()
    out = model(**inp)
    loss = out.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(float(loss.detach().cpu()))
    info(f"  step {step}: loss = {losses[-1]:.4f}")


info("\n" + "=" * 60)
info(f"SMOKE TEST COMPLETE")
info(f"  initial loss: {losses[0]:.4f}")
info(f"  final loss:   {losses[-1]:.4f}")
info(f"  loss decreased: {losses[0] > losses[-1]}")
info("=" * 60)

with open(OUT_REPORT, "w") as f:
    json.dump({
        "model": BASE_MODEL,
        "n_samples": len(samples),
        "losses": losses,
        "loss_decreased": losses[0] > losses[-1],
        "vision_target_modules": len(vision_targets),
    }, f, indent=2)
info(f"  saved {OUT_REPORT}")
