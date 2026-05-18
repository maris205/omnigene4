#!/usr/bin/env python
# coding: utf-8
"""
43-eval_v5_full.py

SFT v5 全面评测:
  1. Multi-task generation (6 categories x 100 samples) - 验证无 regression
  2. Standard / Remote homology - 验证无 regression
  3. BixBench - 验证无 regression
  4. Structure: generation mode vs classification head mode 对比
     - Generation: 模型生成 <SEQ_3Di>...</SEQ_3Di>, 提取后比较
     - Classifier: 用 head_3di / head_dssp 在 prompt 后续位置直接分类

预计 1.5-2 小时.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import torch
import torch.nn as nn
import json
import random
import re
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, inject_adapter_in_model
from datasets import load_dataset

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_V5_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-sft-classifier"
EVAL_FILE = "/root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl"
OUT_JSON = "/root/autodl-tmp/dnagpt/outputs/v5_full_eval.json"

N_PER_CAT = 100
N_STD_HOM = 1000
N_REM_HOM = 500
N_STRUCT_CLS = 50  # classifier mode 测试样本数
SEED = 42
HIDDEN_SIZE = 2816
N_3DI = 20
N_DSSP = 8

TDI_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
DSSP_ALPHABET = list("HBEGITSC")
TDI_INV = {i: c for i, c in enumerate(TDI_ALPHABET)}
DSSP_INV = {i: c for i, c in enumerate(DSSP_ALPHABET)}

random.seed(SEED)


# ============== 加载 v5 (LoRA + classifier heads) ==============
print("Loading SFT v5...", flush=True)
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
v5_lora = torch.load(f"{SFT_V5_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
for k, v in v5_lora.items():
    if k in ms:
        ms[k].copy_(v); loaded += 1
print(f"  Loaded {loaded} v5 LoRA tensors")

v5_embed = torch.load(f"{SFT_V5_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(v5_embed)
model.eval()

# Load classification heads
heads = torch.load(f"{SFT_V5_DIR}/struct_heads.pt", map_location="cpu")
head_3di = nn.Linear(HIDDEN_SIZE, N_3DI).to(torch.bfloat16).cuda()
head_dssp = nn.Linear(HIDDEN_SIZE, N_DSSP).to(torch.bfloat16).cuda()
head_3di.load_state_dict(heads["head_3di"])
head_dssp.load_state_dict(heads["head_dssp"])
head_3di.eval()
head_dssp.eval()
print(f"  Loaded classification heads (3Di:20, DSSP:8)")

tokenizer = AutoTokenizer.from_pretrained(SFT_V5_DIR)


# ============== Helpers ==============
def make_prompt(instruction, inp):
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


def task_of(instr):
    s = instr.lower()
    if "3di" in s or "foldseek" in s:
        return "3Di"
    if "secondary structure" in s or "dssp" in s:
        return "DSSP"
    return "Other"


def extract_struct(output):
    m = re.search(r'<SEQ_3Di>([A-Z]+)</SEQ_3Di>', output)
    if m:
        return ("3Di", m.group(1))
    m = re.search(r'<SEQ_2D>([A-Z]+)</SEQ_2D>', output)
    if m:
        return ("DSSP", m.group(1))
    return (None, None)


# ============== 1. Multi-task ==============
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
    structure_chars = []
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
            unique_chars = len(set(re.sub(r'[^A-Z]', '', pred.upper())))
            structure_chars.append(unique_chars)

    avg = sum(scores) / len(scores) if scores else 0
    print(f"  {cat}: keyword score = {avg:.3f}")

    mt_results[cat] = {
        "n": len(scores),
        "keyword_score": avg,
        "char_overlap": sum(char_scores)/len(char_scores) if char_scores else None,
        "avg_unique_chars": sum(structure_chars)/len(structure_chars) if structure_chars else None,
    }


# ============== 2. Classification head Structure 评测 ==============
print("\n========== 2. Classification head on Structure ==========", flush=True)

# 抽 N_STRUCT_CLS 条 Structure 数据
struct_items = [r for r in all_data if r.get("category") == "Structure"]
struct_samples = random.sample(struct_items, min(N_STRUCT_CLS, len(struct_items)))

cls_results = {"3Di": {"per_residue_acc": [], "n": 0},
               "DSSP": {"per_residue_acc": [], "n": 0}}

for i, item in enumerate(struct_samples):
    if i % 10 == 0:
        print(f"  cls {i}/{len(struct_samples)}", flush=True)

    kind, ref_seq = extract_struct(item["output"])
    if not kind or not ref_seq:
        continue

    instruction = item["instruction"]
    inp = item.get("input", "")
    prompt = make_prompt(instruction, inp)

    # 把 reference 序列也拼到 prompt 后面（teacher forcing），用 hidden 推断 target id
    # 这样不受 generation 失败影响，纯测 classifier head 的能力
    full_text = prompt + ref_seq
    ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1500).input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).input_ids
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]  # [seq_len, H]

    target_alphabet = TDI_ALPHABET if kind == "3Di" else DSSP_ALPHABET
    target_inv = TDI_INV if kind == "3Di" else DSSP_INV
    target_to_id = {c: i for i, c in enumerate(target_alphabet)}
    head = head_3di if kind == "3Di" else head_dssp

    # 找 ref_seq 中每个字符在 ids 中的位置
    correct = 0; total = 0
    for j, tid in enumerate(ids[0].cpu().tolist()):
        if j < prompt_len:
            continue
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if len(tok_str) == 1 and tok_str.upper() in target_to_id:
            true_id = target_to_id[tok_str.upper()]
            with torch.no_grad():
                logits = head(hidden[j])
                pred_id = logits.argmax().item()
            if pred_id == true_id:
                correct += 1
            total += 1

    if total > 0:
        cls_results[kind]["per_residue_acc"].append(correct / total)
        cls_results[kind]["n"] += 1

# 汇总
for k in ["3Di", "DSSP"]:
    accs = cls_results[k]["per_residue_acc"]
    if accs:
        cls_results[k]["mean_acc"] = sum(accs) / len(accs)
        cls_results[k]["chance"] = 1.0 / (N_3DI if k == "3Di" else N_DSSP)
    else:
        cls_results[k]["mean_acc"] = 0
print(f"\n=== Classification head accuracy ===")
print(f"  3Di:  {cls_results['3Di'].get('mean_acc', 0):.3f} (chance: {1/N_3DI:.3f}) on {cls_results['3Di']['n']} samples")
print(f"  DSSP: {cls_results['DSSP'].get('mean_acc', 0):.3f} (chance: {1/N_DSSP:.3f}) on {cls_results['DSSP']['n']} samples")


# ============== 3. Standard Homology ==============
print("\n========== 3. Standard Homology ==========", flush=True)
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


# ============== 4. Remote Homology ==============
print("\n========== 4. Remote Homology ==========", flush=True)
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


# ============== 5. BixBench ==============
print("\n========== 5. BixBench ==========", flush=True)
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
    "model": "OmniGene-4-SFT-v5 (4-bit + Alpaca + classification heads)",
    "multi_task": mt_results,
    "classification_heads": cls_results,
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
print("Multi-task (generation):")
for cat, r in mt_results.items():
    line = f"{cat:<14s}: keyword {r['keyword_score']:.3f}"
    if r.get('char_overlap') is not None:
        line += f", char {r['char_overlap']:.3f}, unique-chars {r['avg_unique_chars']:.1f}"
    print(f"  {line}")
print()
print("Classification heads (per-residue):")
print(f"  3Di:  {cls_results['3Di'].get('mean_acc', 0):.3f} (chance: {1/N_3DI:.3f})")
print(f"  DSSP: {cls_results['DSSP'].get('mean_acc', 0):.3f} (chance: {1/N_DSSP:.3f})")
print(f"\nSaved {OUT_JSON}")
