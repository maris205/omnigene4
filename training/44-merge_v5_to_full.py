#!/usr/bin/env python
# coding: utf-8
"""
44-merge_v5_to_full.py

合并 SFT v5 LoRA + embedding + struct_heads 到 base model.
输出: 完整 BF16 模型 (~50GB) + struct_heads.pt 单独文件
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import shutil

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_V5_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-sft-classifier"
V5_MERGED_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"

print("=" * 80)
print("合并 SFT v5 LoRA + embedding 到 base (BF16)")
print("=" * 80)

# 1. 加载 base model BF16
print("\n[1/5] 加载 Base Model (BF16)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"  Base model loaded")

# 2. 手动合并 v5 LoRA
print("\n[2/5] 手动合并 v5 LoRA...", flush=True)
v5_lora = torch.load(f"{SFT_V5_DIR}/lora_weights.pt", map_location="cpu")
print(f"  Loaded {len(v5_lora)} LoRA tensors")

merged_count = 0
scaling = 128.0 / 64.0  # alpha / r
for name, param in model.named_parameters():
    lora_A_name = name.replace(".weight", ".lora_A.weight")
    lora_B_name = name.replace(".weight", ".lora_B.weight")
    if lora_A_name in v5_lora and lora_B_name in v5_lora:
        lora_A = v5_lora[lora_A_name].to(param.device).to(param.dtype)
        lora_B = v5_lora[lora_B_name].to(param.device).to(param.dtype)
        delta_weight = (lora_B @ lora_A) * scaling
        param.data += delta_weight
        merged_count += 1
print(f"  Merged {merged_count} v5 LoRA layers")

# 3. 加载 v5 embedding
print("\n[3/5] 加载 v5 embedding...", flush=True)
v5_embed = torch.load(f"{SFT_V5_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(
    v5_embed.to(model.get_input_embeddings().weight.dtype).to(model.get_input_embeddings().weight.device)
)
print(f"  Loaded v5 embedding ({v5_embed.shape})")

# 4. 保存合并模型
print(f"\n[4/5] 保存到 {V5_MERGED_DIR}...", flush=True)
Path(V5_MERGED_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(V5_MERGED_DIR, safe_serialization=True, max_shard_size="5GB")

# 拷贝 tokenizer + meta + struct_heads (v5 特有)
tokenizer = AutoTokenizer.from_pretrained(SFT_V5_DIR)
tokenizer.save_pretrained(V5_MERGED_DIR)
shutil.copy(f"{SFT_V5_DIR}/struct_heads.pt", f"{V5_MERGED_DIR}/struct_heads.pt")
if Path(f"{SFT_V5_DIR}/bio_sft_v5_meta.json").exists():
    shutil.copy(f"{SFT_V5_DIR}/bio_sft_v5_meta.json", f"{V5_MERGED_DIR}/bio_sft_v5_meta.json")
print(f"  v5 merged model + struct_heads saved")

# 5. 统计
print("\n[5/5] 统计模型大小...", flush=True)
import subprocess
size = subprocess.check_output(f"du -sh {V5_MERGED_DIR}", shell=True).decode().split()[0]
print(f"\n{'=' * 80}")
print(f"合并完成! v5 merged: {V5_MERGED_DIR} ({size})")
print(f"{'=' * 80}")
print(f"\n包含:")
print(f"  - model-*.safetensors (BF16, ~50GB)")
print(f"  - struct_heads.pt (3Di + DSSP classifiers, 157KB)")
print(f"  - tokenizer + chat template")
print(f"\n上传命令:")
print(f"  hf upload dnagpt/OmniGene-4-SFT-v5-merged {V5_MERGED_DIR} . --repo-type model")
