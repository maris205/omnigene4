#!/usr/bin/env python
# coding: utf-8
"""
60-eval_stage2.py

Evaluate OmniGene-4-MM Stage 2 on:
1. Vis-CheBI20 test (50 per task type) -- did vision survive Stage 2?
2. BioPAWS protein_pair_short (200 pairs) -- did standard homology recover?
3. BioPAWS protein_pair_remote (200 pairs) -- carryover (high bar)
4. Multi-task generation (cell, mol, protein 各 50) -- text reasoning recovered?

Compare against Stage 1 to see whether forgetting was undone.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import re
import random
import torch
from PIL import Image
from collections import defaultdict
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, inject_adapter_in_model
from datasets import load_dataset

BASE_MODEL = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR     = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3"
CHEBI_TEST = "/root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20/test.json"
CHEBI_BASE = "/root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20"
EVAL_FILE  = "/root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl"
OUT_REPORT = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3/eval_report.json"

random.seed(42)

print("=" * 60)
print("OmniGene-4-MM Stage 2 evaluation")
print("=" * 60)

# ============== Load model ==============
print("\n[1/5] Loading...", flush=True)
processor = AutoProcessor.from_pretrained(MM_DIR)
tokenizer = AutoTokenizer.from_pretrained(MM_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
lora_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj','router.proj'],
)
inject_adapter_in_model(lora_cfg, model.model.language_model, adapter_name="stage3")
ms = model.state_dict()
mm_lora = torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
for k, v in mm_lora.items():
    if k in ms: ms[k].copy_(v); loaded += 1
print(f"  loaded {loaded} LoRA tensors")

mm_emb = torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(mm_emb)
model.eval()
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ============== Helpers ==============
def gen_with_image(image_path, user_prompt, max_new_tokens=80):
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": user_prompt},
    ]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = processor(text=text, images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id or 0,
        )
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


def gen_text(prompt, max_new_tokens=8):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id or 0,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def parse_homology(text):
    t = text.strip().lower()[:40]
    if 'non-homolog' in t or 'non homolog' in t or 'nonhomolog' in t: return 0
    if 'homolog' in t or 'yes' in t: return 1
    if 'no' in t: return 0
    return None


def make_alpaca_prompt(instruction, inp=""):
    if inp and inp.strip():
        return f"### Instruction:\n{instruction}\n\n{inp}\n\n### Answer:\n"
    return f"### Instruction:\n{instruction}\n\n### Answer:\n"


def normalize_text(t):
    return re.sub(r'[^a-z0-9 ]', ' ', t.lower())


def keyword_score(pred, ref, k=5):
    p = normalize_text(pred); r = normalize_text(ref)
    words = sorted(set(w for w in r.split() if len(w) >= 4), key=len, reverse=True)[:k]
    if not words: return None
    return sum(1 for w in words if w in p) / len(words)


# ============== 2. Vis-CheBI20 ==============
print("\n[2/5] Vis-CheBI20 (50 per task)...", flush=True)
with open(CHEBI_TEST) as f:
    chebi = json.load(f)
by_task = defaultdict(list)
for r in chebi: by_task[r["task_name"]].append(r)

chebi_results = {}
for task_name, items in by_task.items():
    samples = random.sample(items, min(50, len(items)))
    correct = 0; n = 0; overlap_sum = 0
    for i, r in enumerate(samples):
        if i % 10 == 0: print(f"  {task_name}: {i}/{len(samples)}", flush=True)
        rel_img = r["images"][0]
        img_path = f"{CHEBI_BASE}/{rel_img}"
        if not os.path.exists(img_path): continue
        user = r["messages"][0]["content"].replace("<image>", "").strip()
        ref = r["messages"][1]["content"]
        try:
            pred = gen_with_image(img_path, user, max_new_tokens=120)
        except Exception:
            continue
        ref_clean = ref.replace("The SMILES is", "").replace("The IUPAC is", "").strip().rstrip('.')
        pred_clean = pred.replace("The SMILES is", "").replace("The IUPAC is", "").strip().rstrip('.')
        if task_name in ("trans_smiles", "trans_iupac"):
            exact = (re.sub(r'\s+', '', pred_clean) == re.sub(r'\s+', '', ref_clean))
            correct += int(exact)
            p = re.sub(r'[^A-Za-z0-9]', '', pred_clean)
            r_ = re.sub(r'[^A-Za-z0-9]', '', ref_clean)
            L = min(len(p), len(r_))
            if L > 0:
                overlap_sum += sum(1 for j in range(L) if p[j] == r_[j]) / L
        else:
            ref_w = set(re.findall(r'\w{4,}', ref_clean.lower()))
            pred_w = set(re.findall(r'\w{4,}', pred_clean.lower()))
            if ref_w:
                ov = len(ref_w & pred_w) / len(ref_w)
                correct += int(ov >= 0.3)
                overlap_sum += ov
        n += 1
    chebi_results[task_name] = {
        "n": n, "acc": correct / max(n,1), "overlap": overlap_sum / max(n,1),
    }
    print(f"  [{task_name}] n={n}, acc={correct/max(n,1):.3f}, overlap={overlap_sum/max(n,1):.3f}")


# ============== 3. Standard homology ==============
print("\n[3/5] Standard homology (200 pairs)...", flush=True)

def hom_prompt(s1, s2):
    return ("### Instruction:\nDetermine if the two sequences below are "
            "structurally related (like paraphrases).\n\n"
            f"### Sequence 1:\n{s1}\n\n### Sequence 2:\n{s2}\n\n### Answer:\n")

ds_std = load_dataset("dnagpt/biopaws", "protein_pair_short", split="train")
data0 = [x for x in ds_std if int(x["label"]) == 0]
data1 = [x for x in ds_std if int(x["label"]) == 1]
random.seed(42)
std_samples = random.sample(data0, 100) + random.sample(data1, 100)
random.shuffle(std_samples)
std_correct = 0; std_valid = 0
for i, p in enumerate(std_samples):
    if i % 50 == 0: print(f"  std {i}/200", flush=True)
    resp = gen_text(hom_prompt(p["sentence1"], p["sentence2"]), max_new_tokens=8)
    pred = parse_homology(resp)
    if pred is not None:
        std_valid += 1
        if pred == int(p["label"]): std_correct += 1
std_acc = std_correct / max(std_valid, 1)
print(f"  Standard: {std_acc:.4f} ({std_correct}/{std_valid})")


# ============== 4. Remote homology ==============
print("\n[4/5] Remote homology (200 pairs)...", flush=True)
ds_rem = load_dataset("dnagpt/biopaws", "protein_pair_remote", split="train")
data0 = [x for x in ds_rem if int(x["label"]) == 0]
data1 = [x for x in ds_rem if int(x["label"]) == 1]
random.seed(42)
rem_samples = random.sample(data0, 100) + random.sample(data1, 100)
random.shuffle(rem_samples)
rem_correct = 0; rem_valid = 0
for i, p in enumerate(rem_samples):
    if i % 50 == 0: print(f"  rem {i}/200", flush=True)
    resp = gen_text(hom_prompt(p["sentence1"], p["sentence2"]), max_new_tokens=8)
    pred = parse_homology(resp)
    if pred is not None:
        rem_valid += 1
        if pred == int(p["label"]): rem_correct += 1
rem_acc = rem_correct / max(rem_valid, 1)
print(f"  Remote: {rem_acc:.4f} ({rem_correct}/{rem_valid})")


# ============== 5. Multi-task gen (sample 30 each) ==============
print("\n[5/5] Multi-task generation (30 per category)...", flush=True)
all_eval = []
if os.path.exists(EVAL_FILE):
    with open(EVAL_FILE) as f:
        for line in f: all_eval.append(json.loads(line))
print(f"  loaded {len(all_eval)} eval rows")

by_cat = defaultdict(list)
for r in all_eval:
    by_cat[r.get("category", "unknown")].append(r)

mt_results = {}
for cat in ("Cell", "Mol", "Protein", "Literature"):
    items = by_cat.get(cat, [])
    if not items: continue
    samples = random.sample(items, min(30, len(items)))
    scores = []
    for i, item in enumerate(samples):
        prompt = make_alpaca_prompt(item["instruction"], item.get("input", ""))
        try:
            pred = gen_text(prompt, max_new_tokens=200).strip()
            kw = keyword_score(pred, item["output"])
            if kw is not None: scores.append(kw)
        except Exception: continue
    avg = sum(scores) / max(len(scores), 1)
    mt_results[cat] = {"n": len(scores), "kw_score": avg}
    print(f"  {cat}: kw_score={avg:.3f} (n={len(scores)})")


# ============== Save ==============
report = {
    "model": "OmniGene-4-MM-stage3",
    "vis_chebi20": chebi_results,
    "standard_homology": {"acc": std_acc, "valid": std_valid, "correct": std_correct,
                          "v5_baseline": 0.994},
    "remote_homology": {"acc": rem_acc, "valid": rem_valid, "correct": rem_correct,
                        "v5_baseline": 0.826},
    "multi_task_gen": mt_results,
}
with open(OUT_REPORT, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved {OUT_REPORT}")

# Compare to Stage 2
print("\n" + "=" * 60)
print("COMPARISON: Stage 2 -> Stage 3")
print("=" * 60)
stage_prev_path = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2/eval_report.json"
if os.path.exists(stage_prev_path):
    s_prev = json.load(open(stage_prev_path))
    for task in chebi_results:
        s_prev_acc = s_prev["vis_chebi20"][task]["acc"]
        s3_acc = chebi_results[task]["acc"]
        delta = s3_acc - s_prev_acc
        sign = "+" if delta >= 0 else ""
        print(f"  vis_{task:14s}: S2 {s_prev_acc:.3f} -> S3 {s3_acc:.3f} ({sign}{delta:+.3f})")
    print(f"  remote_homology   : S2 {s_prev['remote_homology']['acc']:.3f} -> S3 {rem_acc:.3f}")
    print(f"  standard_homology : S2 {s_prev['standard_homology']['acc']:.3f} -> S3 {std_acc:.3f}")
print(f"  v5 baseline       : standard 0.994, remote 0.826")
