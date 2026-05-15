#!/usr/bin/env python
# coding: utf-8
"""
37-eval_structure_bf16.py

用 BF16 完整模型测 Structure 任务，看 4-bit 失败的根因是什么。

测试假设:
  - 假设 A: 4-bit 量化破坏了 Structure 任务 → BF16 应该 work
  - 假设 B: 训练数据/方法本身不够 → BF16 也会失败

每个测试 50 个样本 (Structure category)。
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import torch
import json
import random
import re
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# 用 BF16 完整 merged 模型 (不量化)
MODEL_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-SFT-v3-merged"
EVAL_FILE = "/root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl"
OUT_JSON = "/root/autodl-tmp/dnagpt/outputs/structure_bf16_eval.json"

N_SAMPLES = 50
MAX_NEW_TOKENS = 250  # Structure 序列较长
SEED = 42

random.seed(SEED)

print("Loading OmniGene-4 v3 (BF16 完整, 不量化)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()


def make_prompt(instruction, inp):
    if inp.strip():
        return (
            "<User>\n### Instruction:\n" + instruction + "\n\n"
            + inp + "\n### Answer:\n<Assistant>\n"
        )
    return (
        "<User>\n### Instruction:\n" + instruction + "\n\n"
        "### Answer:\n<Assistant>\n"
    )


def generate(prompt, max_tokens=MAX_NEW_TOKENS):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def collapse_score(text):
    """字符多样性分: 越多样越接近 1, 全是单字符 (collapse) 接近 0"""
    t = re.sub(r'[^A-Z]', '', text.upper())
    if len(t) < 5:
        return 0.0
    n_unique = len(set(t))
    return n_unique / 20.0  # 最多 20 个 3Di / 8 个 DSSP, 归一化


def char_overlap(pred, ref):
    """字符级重叠 (3Di/DSSP 是单字符序列)"""
    p = re.sub(r'[^A-Z]', '', pred.upper())
    r = re.sub(r'[^A-Z]', '', ref.upper())
    if not p or not r:
        return 0.0
    L = min(len(p), len(r))
    if L == 0:
        return 0.0
    matches = sum(1 for i in range(L) if p[i] == r[i])
    return matches / L


# 读 eval 集
print(f"\nLoading eval set...", flush=True)
all_data = []
with open(EVAL_FILE) as f:
    for line in f:
        r = json.loads(line)
        if r.get("category") == "Structure":
            all_data.append(r)

print(f"  Structure samples available: {len(all_data)}")
samples = random.sample(all_data, min(N_SAMPLES, len(all_data)))
print(f"  Testing on: {len(samples)}")

# 评测
results = []
for i, item in enumerate(samples):
    if i % 10 == 0:
        print(f"  {i}/{len(samples)}", flush=True)
    prompt = make_prompt(item["instruction"], item.get("input", ""))
    try:
        pred = generate(prompt).strip()
    except Exception as e:
        print(f"    error: {e}")
        continue

    ref = item["output"].strip()
    div = collapse_score(pred)
    overlap = char_overlap(pred, ref)
    results.append({
        "instruction": item["instruction"][:80],
        "input": item.get("input", "")[:80],
        "ref": ref[:150],
        "pred": pred[:150],
        "diversity": div,
        "char_overlap": overlap,
    })

# 统计
divs = [r["diversity"] for r in results]
overlaps = [r["char_overlap"] for r in results]
collapse_count = sum(1 for r in results if r["diversity"] < 0.15)

print(f"\n=== BF16 Structure Task Results (n={len(results)}) ===")
print(f"  Avg char-level overlap with ref: {sum(overlaps)/len(overlaps):.3f}")
print(f"  Avg output diversity (alphabet usage): {sum(divs)/len(divs):.3f}")
print(f"  Collapse rate (diversity < 0.15): {collapse_count}/{len(results)} = {100*collapse_count/len(results):.0f}%")

print("\n=== Sample predictions ===")
for s in results[:5]:
    print(f"\n[diversity={s['diversity']:.2f}, overlap={s['char_overlap']:.2f}]")
    print(f"  REF:  {s['ref'][:100]}")
    print(f"  PRED: {s['pred'][:100]}")

# 保存
with open(OUT_JSON, "w") as f:
    json.dump({
        "model": "OmniGene-4-SFT-v3 (BF16 unquantized)",
        "task": "Structure (3Di/DSSP prediction)",
        "n_samples": len(results),
        "avg_char_overlap_with_ref": sum(overlaps)/len(overlaps) if overlaps else 0,
        "avg_diversity": sum(divs)/len(divs) if divs else 0,
        "collapse_rate": collapse_count / len(results) if results else 0,
        "samples": results,
    }, f, indent=2)

print(f"\nSaved {OUT_JSON}")
