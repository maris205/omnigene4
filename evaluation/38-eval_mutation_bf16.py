#!/usr/bin/env python
# coding: utf-8
"""
38-eval_mutation_bf16.py

用 BF16 完整模型测 Mutation 任务，对照 4-bit 的 10.8%。
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import torch
import json
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-SFT-v3-merged"
EVAL_FILE = "/root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl"
OUT_JSON = "/root/autodl-tmp/dnagpt/outputs/mutation_bf16_eval.json"

N_SAMPLES = 50
MAX_NEW_TOKENS = 200
SEED = 42
random.seed(SEED)

print("Loading BF16 v3...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()

def make_prompt(instruction, inp):
    if inp.strip():
        return f"<User>\n### Instruction:\n{instruction}\n\n{inp}\n### Answer:\n<Assistant>\n"
    return f"<User>\n### Instruction:\n{instruction}\n\n### Answer:\n<Assistant>\n"

def generate(prompt):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def normalize(t):
    return re.sub(r'[^a-z0-9 ]', ' ', t.lower())


def keyword_score(pred, ref, k=5):
    p = normalize(pred); r = normalize(ref)
    words = sorted(set(w for w in r.split() if len(w) >= 4), key=len, reverse=True)[:k]
    if not words: return None
    return sum(1 for w in words if w in p) / len(words)


# 读 Mutation
all_data = []
with open(EVAL_FILE) as f:
    for line in f:
        r = json.loads(line)
        if r.get("category") == "Mutation":
            all_data.append(r)
print(f"Mutation samples: {len(all_data)}")
samples = random.sample(all_data, min(N_SAMPLES, len(all_data)))
print(f"Testing on: {len(samples)}")

results = []
generic_count = 0  # 看模型有多少回答是 generic / 不指向具体突变效应
for i, item in enumerate(samples):
    if i % 10 == 0:
        print(f"  {i}/{len(samples)}", flush=True)
    prompt = make_prompt(item["instruction"], item.get("input", ""))
    try:
        pred = generate(prompt).strip()
    except Exception as e:
        continue

    ref = item["output"].strip()
    score = keyword_score(pred, ref)

    # 检测 generic 回答
    pred_low = pred.lower()
    is_generic = any(g in pred_low for g in [
        "in a breast cancer sample", "somatic mutation", "no specific",
        "abolishes interaction with",
    ]) and len(pred) < 80

    if is_generic:
        generic_count += 1

    results.append({
        "instruction": item["instruction"][:100],
        "ref": ref[:200],
        "pred": pred[:200],
        "score": score if score is not None else 0,
        "is_generic": is_generic,
    })

avg = sum(r["score"] for r in results) / len(results)
print(f"\n=== Mutation BF16 (n={len(results)}) ===")
print(f"  Avg keyword score: {avg:.3f}")
print(f"  Generic answers: {generic_count}/{len(results)} = {100*generic_count/len(results):.0f}%")

print("\n=== Sample predictions ===")
for r in results[:5]:
    print(f"\n[score={r['score']:.2f}, generic={r['is_generic']}]")
    print(f"  REF:  {r['ref'][:120]}")
    print(f"  PRED: {r['pred'][:120]}")

with open(OUT_JSON, "w") as f:
    json.dump({
        "model": "OmniGene-4-SFT-v3 (BF16 unquantized)",
        "task": "Mutation effect description",
        "n_samples": len(results),
        "avg_keyword_score": avg,
        "generic_rate": generic_count / len(results),
        "samples": results[:30],
    }, f, indent=2)

print(f"\nSaved {OUT_JSON}")
