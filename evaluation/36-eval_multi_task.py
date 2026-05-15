#!/usr/bin/env python
# coding: utf-8
"""
36-eval_multi_task.py

跑 SFT v3 在 6 个任务类别上的评测：
  Literature, Protein, Mutation, Cell, Structure, Mol

eval set: /root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl
每个类别采样 N_PER_CAT 条做评测。

精度等价 GGUF Q4_K_M（同样是 NF4 4-bit）。
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, inject_adapter_in_model

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v3-sft-remote"
EVAL_FILE = "/root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl"
OUT_JSON = "/root/autodl-tmp/dnagpt/outputs/multi_task_eval.json"

N_PER_CAT = 150
MAX_NEW_TOKENS = 200
SEED = 42

random.seed(SEED)

print("Loading OmniGene-4 v3 (4-bit NF4)...", flush=True)
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map={"": 0},
)
lora_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
inject_adapter_in_model(lora_config, model.model.language_model, adapter_name="default")
ms = model.state_dict()
sft_lora = torch.load(f"{SFT_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
for k, v in sft_lora.items():
    if k in ms:
        ms[k].copy_(v); loaded += 1
print(f"  Loaded {loaded} SFT LoRA tensors")
sft_embed = torch.load(f"{SFT_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(sft_embed)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(SFT_DIR)


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


def normalize(text):
    """简单归一化用于 keyword 匹配"""
    return re.sub(r'[^a-z0-9 ]', ' ', text.lower())


def keyword_overlap_score(pred, ref, top_k=5):
    """生成式任务的简单评分: 取参考答案前 top_k 关键词，看预测是否包含"""
    p = normalize(pred)
    r = normalize(ref)
    # 用空格切分，按长度排，长词更具区分力
    ref_words = sorted(set(w for w in r.split() if len(w) >= 4), key=len, reverse=True)[:top_k]
    if not ref_words:
        return None
    hits = sum(1 for w in ref_words if w in p)
    return hits / len(ref_words)


# 读 eval 集
print(f"\nLoading eval set from {EVAL_FILE}", flush=True)
all_data = []
with open(EVAL_FILE) as f:
    for line in f:
        all_data.append(json.loads(line))
print(f"  Total: {len(all_data)}")

# 按 category 分组
by_cat = defaultdict(list)
for r in all_data:
    by_cat[r.get("category", "unknown")].append(r)

print(f"  Categories: {[(c, len(v)) for c, v in by_cat.items()]}")

# 每个 category 随机采样 N_PER_CAT
results = {}
for cat in sorted(by_cat.keys()):
    items = by_cat[cat]
    n = min(N_PER_CAT, len(items))
    samples = random.sample(items, n)
    print(f"\n=== {cat} ({n} samples) ===", flush=True)

    scores = []
    failed = 0
    samples_out = []
    for i, item in enumerate(samples):
        if i % 30 == 0:
            print(f"  {cat} {i}/{n}", flush=True)
        prompt = make_prompt(item["instruction"], item.get("input", ""))
        try:
            pred = generate(prompt).strip()
        except Exception as e:
            print(f"    error: {e}")
            failed += 1
            continue

        ref = item["output"].strip()
        score = keyword_overlap_score(pred, ref)
        if score is not None:
            scores.append(score)
        if i < 3:  # 留 3 个样本看效果
            samples_out.append({
                "instruction": item["instruction"][:100],
                "input": item.get("input", "")[:100],
                "ref": ref[:200],
                "pred": pred[:200],
                "score": score,
            })

    if scores:
        avg = sum(scores) / len(scores)
        print(f"  {cat}: avg score = {avg:.3f} ({len(scores)}/{n} valid, {failed} failed)")
    else:
        avg = 0.0
        print(f"  {cat}: no valid scores")

    results[cat] = {
        "n_samples": n,
        "n_valid": len(scores),
        "n_failed": failed,
        "avg_keyword_overlap": avg,
        "samples": samples_out,
    }

# 总体得分
all_scores = []
for cat, r in results.items():
    all_scores.extend([r["avg_keyword_overlap"]] * r["n_valid"])

overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

with open(OUT_JSON, "w") as f:
    json.dump({
        "model": "OmniGene-4-SFT-v3 (4-bit NF4 ≡ GGUF Q4_K_M precision)",
        "n_per_category": N_PER_CAT,
        "overall_avg_score": overall,
        "by_category": results,
    }, f, indent=2)

print(f"\n=== Overall: {overall:.3f} ===")
print(f"Saved {OUT_JSON}")
print("\n=== Summary by category ===")
for cat in sorted(results.keys()):
    r = results[cat]
    print(f"  {cat:<14s}: {r['avg_keyword_overlap']:.3f} ({r['n_valid']}/{r['n_samples']})")
