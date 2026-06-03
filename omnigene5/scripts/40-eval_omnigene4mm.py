#!/usr/bin/env python
# coding: utf-8
"""
40-eval_omnigene4mm.py

Evaluate OmniGene-4-MM (Stage 1) on:
1. Vis-CheBI20 test split: image -> SMILES (50 samples per task type)
2. Carryover: protein_pair_remote 200 pairs (verify text path not collapsed)

Output: /root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage1/eval_report.json
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import re
import random
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, inject_adapter_in_model
from datasets import load_dataset

BASE_MODEL = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR     = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage1"
TEST_JSON  = "/root/autodl-tmp/dnagpt/data/B_chebi20_redirect/test.json"  # fallback below
OUT_REPORT = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage1/eval_report.json"

# Vis-CheBI20 test
CHEBI_TEST = "/root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20/test.json"
CHEBI_BASE = "/root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20"

random.seed(42)

print("=" * 60)
print("OmniGene-4-MM evaluation")
print("=" * 60)


# ============== Load model + LoRA ==============
print("\n[1/4] Loading OmniGene-4-MM (BF16 base + Stage 1 LoRA)...", flush=True)
processor = AutoProcessor.from_pretrained(MM_DIR)
tokenizer = AutoTokenizer.from_pretrained(MM_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)

# Inject the same LoRA config
lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    target_modules=['q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj','router.proj'],
)
inject_adapter_in_model(lora_cfg, model.model.language_model, adapter_name="omnigene5")

# Load Stage 1 LoRA weights
ms = model.state_dict()
mm_lora = torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
for k, v in mm_lora.items():
    if k in ms:
        ms[k].copy_(v); loaded += 1
print(f"  loaded {loaded} LoRA tensors", flush=True)

# Load Stage 1 embedding
mm_emb = torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(mm_emb)
print(f"  embedding loaded ({mm_emb.shape})", flush=True)

model.eval()
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


# ============== Helpers ==============
def generate_with_image(image_path, user_prompt, max_new_tokens=80):
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


def generate_text(prompt, max_new_tokens=8):
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


def extract_smiles(text):
    # Look for "The SMILES is X" or just X
    m = re.search(r'SMILES\s*(?:is)?[:\s]*([^\.\n]+)', text, re.IGNORECASE)
    if m: return m.group(1).strip().rstrip('.')
    return text.strip().rstrip('.')


def normalize_smiles(s):
    return re.sub(r'\s+', '', s).strip().rstrip('.').rstrip(',')


# ============== 2. Vis-CheBI20 evaluation ==============
print("\n[2/4] Vis-CheBI20 evaluation (50 per task type)...", flush=True)
with open(CHEBI_TEST) as f:
    chebi_test = json.load(f)

# Group by task type
from collections import defaultdict
by_task = defaultdict(list)
for r in chebi_test:
    by_task[r["task_name"]].append(r)
print(f"  task distribution: { {k: len(v) for k,v in by_task.items()} }")

chebi_results = {}
for task_name, items in by_task.items():
    samples = random.sample(items, min(50, len(items)))
    correct = 0
    char_overlap_sum = 0
    n = 0
    examples = []
    for i, r in enumerate(samples):
        if i % 10 == 0: print(f"  {task_name}: {i}/{len(samples)}", flush=True)
        rel_img = r["images"][0]
        img_path = f"{CHEBI_BASE}/{rel_img}"
        if not os.path.exists(img_path): continue
        user = r["messages"][0]["content"].replace("<image>", "").strip()
        ref = r["messages"][1]["content"]
        try:
            pred = generate_with_image(img_path, user, max_new_tokens=120)
        except Exception as e:
            print(f"    err: {e}")
            continue
        # Compare
        ref_clean = ref.replace("The SMILES is", "").replace("The IUPAC is", "").strip().rstrip('.')
        pred_clean = pred.replace("The SMILES is", "").replace("The IUPAC is", "").strip().rstrip('.')
        if task_name in ("trans_smiles", "trans_iupac"):
            exact = (normalize_smiles(pred_clean) == normalize_smiles(ref_clean))
            correct += int(exact)
            # char overlap
            p = re.sub(r'[^A-Za-z0-9]', '', pred_clean)[:len(ref_clean)]
            r_ = re.sub(r'[^A-Za-z0-9]', '', ref_clean)[:len(pred_clean)]
            L = min(len(p), len(r_))
            if L > 0:
                char_overlap_sum += sum(1 for j in range(L) if p[j] == r_[j]) / L
        else:
            # keyword match for description / functional groups
            ref_words = set(re.findall(r'\w{4,}', ref_clean.lower()))
            pred_words = set(re.findall(r'\w{4,}', pred_clean.lower()))
            if ref_words:
                overlap = len(ref_words & pred_words) / len(ref_words)
                correct += int(overlap >= 0.3)
                char_overlap_sum += overlap
        n += 1
        if i < 3:
            examples.append({"user": user[:80], "ref": ref[:120], "pred": pred[:120]})

    chebi_results[task_name] = {
        "n": n,
        "exact_match_or_keyword_match": correct,
        "accuracy": correct / max(n, 1),
        "avg_char_overlap": char_overlap_sum / max(n, 1),
        "examples": examples[:3],
    }
    print(f"  [{task_name}] n={n}, acc={correct/max(n,1):.4f}, overlap={char_overlap_sum/max(n,1):.3f}")


# ============== 3. Carryover test: protein_pair_remote ==============
print("\n[3/4] Protein remote homology (200 pairs, no image)...", flush=True)
ds = load_dataset("dnagpt/biopaws", "protein_pair_remote", split="train")
data0 = [x for x in ds if int(x["label"]) == 0]
data1 = [x for x in ds if int(x["label"]) == 1]
random.seed(42)
samples = random.sample(data0, 100) + random.sample(data1, 100)
random.shuffle(samples)


def hom_prompt(s1, s2):
    return ("### Instruction:\nDetermine if the two sequences below are "
            "structurally related (like paraphrases).\n\n"
            f"### Sequence 1:\n{s1}\n\n### Sequence 2:\n{s2}\n\n### Answer:\n")


correct, valid = 0, 0
for i, p in enumerate(samples):
    if i % 50 == 0: print(f"  remote {i}/200", flush=True)
    resp = generate_text(hom_prompt(p['sentence1'], p['sentence2']), max_new_tokens=8)
    pred = parse_homology(resp)
    if pred is not None:
        valid += 1
        if pred == int(p['label']): correct += 1

remote_acc = correct / max(valid, 1)
print(f"  Remote homology: {remote_acc:.4f} ({correct}/{valid})")


# ============== 4. Save report ==============
print("\n[4/4] Saving report...", flush=True)
report = {
    "model": "OmniGene-4-MM-stage1",
    "base": BASE_MODEL,
    "lora_dir": MM_DIR,
    "vis_chebi20_test": chebi_results,
    "remote_homology_carryover": {
        "n_pairs": len(samples),
        "valid": valid,
        "correct": correct,
        "acc": remote_acc,
        "v5_baseline": 0.826,
    },
}
with open(OUT_REPORT, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved {OUT_REPORT}")

print("\n=== Summary ===")
for k, v in chebi_results.items():
    print(f"  {k}: acc {v['accuracy']:.3f}, overlap {v['avg_char_overlap']:.3f}")
print(f"  Remote homology: {remote_acc:.3f} (v5 was 0.826)")
