#!/usr/bin/env python
# coding: utf-8
"""
42-eval_v4_full.py

SFT v4 全面评测:
  1. Multi-task (6 categories, 100 each)
  2. Standard Homology (1000 pairs - 抽样, 时间预算)
  3. Remote Homology (500 pairs - 抽样)
  4. BixBench (T/F subset)

关键变化 vs v3 评测:
  - Prompt 改成 Alpaca (与 v4 训练一致, 不再 <User>/<Assistant>)
  - BF16 完整 (不量化, 防止 v3 那种 4-bit collapse 干扰)

预计 2-3 小时.
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
from datasets import load_dataset

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_V4_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v4-sft"
EVAL_FILE = "/root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl"
OUT_JSON = "/root/autodl-tmp/dnagpt/outputs/v4_full_eval.json"

N_PER_CAT = 100
N_STD_HOM = 1000
N_REM_HOM = 500
SEED = 42

random.seed(SEED)


# ============== 加载 BF16 + v4 LoRA ==============
print("Loading SFT v4 (BF16)...", flush=True)
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
v4_lora = torch.load(f"{SFT_V4_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
for k, v in v4_lora.items():
    if k in ms:
        ms[k].copy_(v); loaded += 1
print(f"  Loaded {loaded} LoRA tensors")
v4_embed = torch.load(f"{SFT_V4_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(v4_embed)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(SFT_V4_DIR)


def make_prompt(instruction, inp):
    """v4 训练用的 Alpaca 风格 (不带 chat tag)"""
    if inp and inp.strip():
        return f"### Instruction:\n{instruction}\n\n{inp}\n\n### Answer:\n"
    return f"### Instruction:\n{instruction}\n\n### Answer:\n"


def generate(prompt, max_tokens=200):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1280).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def normalize(t):
    return re.sub(r'[^a-z0-9 ]', ' ', t.lower())


def keyword_score(pred, ref, k=5):
    p = normalize(pred); r = normalize(ref)
    words = sorted(set(w for w in r.split() if len(w) >= 4), key=len, reverse=True)[:k]
    if not words: return None
    return sum(1 for w in words if w in p) / len(words)


def char_overlap(pred, ref):
    p = re.sub(r'[^A-Z]', '', pred.upper())
    r = re.sub(r'[^A-Z]', '', ref.upper())
    if not p or not r: return 0.0
    L = min(len(p), len(r))
    if L == 0: return 0.0
    return sum(1 for i in range(L) if p[i] == r[i]) / L


def parse_homology(text):
    t = text.strip().lower()[:40]
    if 'non-homolog' in t or 'non homolog' in t or 'nonhomolog' in t:
        return 0
    if 'homolog' in t:
        return 1
    if 'yes' in t: return 1
    if 'no' in t: return 0
    return None


# ============== 1. Multi-task eval ==============
print("\n========== 1. Multi-task evaluation ==========", flush=True)
all_data = []
with open(EVAL_FILE) as f:
    for line in f:
        all_data.append(json.loads(line))

by_cat = defaultdict(list)
for r in all_data:
    by_cat[r.get("category", "unknown")].append(r)

mt_results = {}
for cat in sorted(by_cat.keys()):
    items = by_cat[cat]
    n = min(N_PER_CAT, len(items))
    samples = random.sample(items, n)
    print(f"\n=== {cat} (n={n}) ===", flush=True)

    scores = []
    char_scores = []
    structure_chars = []  # for Structure: track collapse
    for i, item in enumerate(samples):
        if i % 25 == 0:
            print(f"  {cat} {i}/{n}", flush=True)
        prompt = make_prompt(item["instruction"], item.get("input", ""))
        try:
            pred = generate(prompt, max_tokens=250).strip()
        except Exception as e:
            print(f"    error: {e}")
            continue

        ref = item["output"].strip()
        kw = keyword_score(pred, ref)
        if kw is not None:
            scores.append(kw)

        if cat == "Structure":
            char_scores.append(char_overlap(pred, ref))
            # 检查是否 collapse
            unique_chars = len(set(re.sub(r'[^A-Z]', '', pred.upper())))
            structure_chars.append(unique_chars)

    avg = sum(scores) / len(scores) if scores else 0
    print(f"  {cat}: keyword score = {avg:.3f}")
    if cat == "Structure" and char_scores:
        avg_char = sum(char_scores) / len(char_scores)
        avg_unique = sum(structure_chars) / len(structure_chars)
        print(f"  Structure char overlap: {avg_char:.3f}, avg unique chars: {avg_unique:.1f}")

    mt_results[cat] = {
        "n": len(scores),
        "keyword_score": avg,
        "char_overlap": sum(char_scores)/len(char_scores) if char_scores else None,
        "avg_unique_chars": sum(structure_chars)/len(structure_chars) if structure_chars else None,
    }


# ============== 2. Standard Homology ==============
print("\n========== 2. Standard Homology ==========", flush=True)
ds = load_dataset('dnagpt/biopaws', 'protein_pair_short')['train']
data0 = [x for x in ds if x['label'] == 0]
data1 = [x for x in ds if x['label'] == 1]
random.seed(SEED)
s0 = random.sample(data0, N_STD_HOM // 2)
s1 = random.sample(data1, N_STD_HOM // 2)
std_data = s0 + s1
random.shuffle(std_data)

def hom_prompt(s1, s2):
    return make_prompt(
        "Determine if the two sequences below are structurally related (like paraphrases).",
        f"### Sequence 1:\n{s1}\n\n### Sequence 2:\n{s2}",
    )

std_correct = 0; std_valid = 0
for i, item in enumerate(std_data):
    if i % 100 == 0:
        print(f"  Std {i}/{len(std_data)}", flush=True)
    resp = generate(hom_prompt(item['sentence1'], item['sentence2']), max_tokens=8)
    pred = parse_homology(resp)
    if pred is not None:
        std_valid += 1
        if pred == item['label']:
            std_correct += 1

std_acc = std_correct / std_valid if std_valid else 0
print(f"Standard: {std_acc:.4f} ({std_correct}/{std_valid})")


# ============== 3. Remote Homology ==============
print("\n========== 3. Remote Homology ==========", flush=True)
rem_ds = load_dataset('dnagpt/biopaws', 'protein_pair_remote')['train']
rd0 = [x for x in rem_ds if int(x['label']) == 0]
rd1 = [x for x in rem_ds if int(x['label']) == 1]
random.seed(SEED)
r0 = random.sample(rd0, N_REM_HOM // 2)
r1 = random.sample(rd1, N_REM_HOM // 2)
rem_data = r0 + r1
random.shuffle(rem_data)

rem_correct = 0; rem_valid = 0
for i, item in enumerate(rem_data):
    if i % 50 == 0:
        print(f"  Rem {i}/{len(rem_data)}", flush=True)
    resp = generate(hom_prompt(item['sentence1'], item['sentence2']), max_tokens=8)
    pred = parse_homology(resp)
    if pred is not None:
        rem_valid += 1
        if pred == int(item['label']):
            rem_correct += 1

rem_acc = rem_correct / rem_valid if rem_valid else 0
print(f"Remote: {rem_acc:.4f} ({rem_correct}/{rem_valid})")


# ============== 4. BixBench ==============
print("\n========== 4. BixBench ==========", flush=True)
bix = load_dataset('futurehouse/BixBench', split='train')
bix_correct = 0; bix_total = 0
for item in bix:
    answer = str(item.get('answer', '')).strip()
    if answer not in ['True', 'False']:
        continue
    h = item.get('hypothesis', '')
    rs = item.get('result', '')
    if not h or not rs:
        continue
    prompt = make_prompt(
        "Based on the research result below, determine if the hypothesis is True or False.\nAnswer only True or False.",
        f"### Hypothesis:\n{h}\n\n### Research Result:\n{rs[:500]}",
    )
    resp = generate(prompt, max_tokens=5).strip().lower()
    pred = 'True' if 'true' in resp[:20] else 'False' if 'false' in resp[:20] else None
    if pred:
        bix_total += 1
        if pred == answer:
            bix_correct += 1

bix_acc = bix_correct / bix_total if bix_total else 0
print(f"BixBench: {bix_acc:.4f} ({bix_correct}/{bix_total})")


# ============== Save ==============
results = {
    "model": "OmniGene-4-SFT-v4 (4-bit NF4 + Alpaca prompt)",
    "multi_task": mt_results,
    "standard_homology": {"acc": std_acc, "correct": std_correct, "valid": std_valid, "n": len(std_data)},
    "remote_homology": {"acc": rem_acc, "correct": rem_correct, "valid": rem_valid, "n": len(rem_data)},
    "bixbench": {"acc": bix_acc, "correct": bix_correct, "total": bix_total},
}
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n========== SUMMARY ==========")
print(f"Standard Homology: {std_acc:.2%}")
print(f"Remote Homology:   {rem_acc:.2%}")
print(f"BixBench T/F:      {bix_acc:.2%}")
print()
for cat, r in mt_results.items():
    line = f"{cat:<14s}: keyword {r['keyword_score']:.3f}"
    if r.get('char_overlap') is not None:
        line += f", char {r['char_overlap']:.3f}, unique-chars {r['avg_unique_chars']:.1f}"
    print(f"  {line}")
print(f"\nSaved {OUT_JSON}")
