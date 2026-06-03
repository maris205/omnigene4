#!/usr/bin/env python
# coding: utf-8
"""
95-qualitative_demo_stage3v3.py

Qualitative showcase of OmniGene-4-MM Stage 3 v3 across all modalities.
Produces a markdown report with prompts + model responses for:
  1. Vis-CheBI20 (3 each: struct_recog, struct_cap, general_desp, trans_iupac, trans_smiles)
  2. Homology (3 std-positive, 2 std-negative, 2 remote-positive, 2 remote-negative)
  3. Multi-task gen (3 each: Cell, Mol, Protein, Literature, Structure)

Output:
  /root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3/qualitative_demo.md
  /root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3/qualitative_demo.json
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import random
import torch
from collections import defaultdict
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, inject_adapter_in_model
from datasets import load_dataset

BASE_MODEL = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR     = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3"
CHEBI_TEST = "/root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20/test.json"
CHEBI_BASE = "/root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20"
EVAL_FILE  = "/root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl"
OUT_MD     = f"{MM_DIR}/qualitative_demo.md"
OUT_JSON   = f"{MM_DIR}/qualitative_demo.json"

random.seed(7)

print("=" * 60)
print("OmniGene-4-MM Stage 3 v3 qualitative showcase")
print("=" * 60)


# ============== Load model ==============
print("\n[1/4] Loading model...", flush=True)
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
inject_adapter_in_model(lora_cfg, model.model.language_model, adapter_name="stage2")
ms = model.state_dict()
loaded = 0
for k, v in torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu").items():
    if k in ms: ms[k].copy_(v); loaded += 1
print(f"  loaded {loaded} LoRA tensors")
model.get_input_embeddings().weight.data.copy_(
    torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu"))
model.eval()
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ============== Helpers ==============
def gen_image(image_path, prompt, max_new=160):
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = processor(text=text, images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


def gen_text(prompt, max_new=200):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def alpaca(instr, inp=""):
    if inp.strip():
        return f"### Instruction:\n{instr}\n\n{inp}\n\n### Answer:\n"
    return f"### Instruction:\n{instr}\n\n### Answer:\n"


results = {"vision": [], "homology": [], "multi_task": []}
print("\n[2/4] Vision (Vis-CheBI20)...", flush=True)
with open(CHEBI_TEST) as f:
    chebi = json.load(f)
by_task = defaultdict(list)
for r in chebi: by_task[r["task_name"]].append(r)

for task in ["struct_recog", "struct_cap", "general_desp", "trans_iupac", "trans_smiles"]:
    picks = random.sample(by_task[task], 3)
    for r in picks:
        img_path = f"{CHEBI_BASE}/{r['images'][0]}"
        if not os.path.exists(img_path): continue
        user = r["messages"][0]["content"].replace("<image>", "").strip()
        ref  = r["messages"][1]["content"]
        try:
            pred = gen_image(img_path, user, max_new=160)
        except Exception as e:
            pred = f"[error: {e}]"
        item = {"task": task, "image": r["images"][0],
                "prompt": user[:300], "reference": ref[:400],
                "prediction": pred.strip()[:600]}
        results["vision"].append(item)
        print(f"  [{task}] image={r['images'][0]}")
        print(f"    pred: {pred.strip()[:100]}")
print(f"  collected {len(results['vision'])} vision items")


print("\n[3/4] Homology (BioPAWS)...", flush=True)
def hom_prompt(s1, s2):
    return ("### Instruction:\nDetermine if the two sequences below are "
            "structurally related (like paraphrases).\n\n"
            f"### Sequence 1:\n{s1}\n\n### Sequence 2:\n{s2}\n\n### Answer:\n")

random.seed(7)
ds_std = load_dataset("dnagpt/biopaws", "protein_pair_short", split="train")
ds_rem = load_dataset("dnagpt/biopaws", "protein_pair_remote", split="train")
std_pos = random.sample([x for x in ds_std if int(x["label"]) == 1], 3)
std_neg = random.sample([x for x in ds_std if int(x["label"]) == 0], 2)
rem_pos = random.sample([x for x in ds_rem if int(x["label"]) == 1], 2)
rem_neg = random.sample([x for x in ds_rem if int(x["label"]) == 0], 2)
hom_picks = [("standard_homologous", p) for p in std_pos] + \
            [("standard_non_homologous", p) for p in std_neg] + \
            [("remote_homologous", p) for p in rem_pos] + \
            [("remote_non_homologous", p) for p in rem_neg]
for kind, p in hom_picks:
    s1 = p["sentence1"]; s2 = p["sentence2"]
    pred = gen_text(hom_prompt(s1, s2), max_new=8).strip()
    item = {"kind": kind, "label": int(p["label"]),
            "seq1_preview": s1[:80] + ("..." if len(s1) > 80 else ""),
            "seq2_preview": s2[:80] + ("..." if len(s2) > 80 else ""),
            "prediction": pred[:80]}
    results["homology"].append(item)
    print(f"  [{kind}] label={p['label']} -> pred='{pred[:30]}'")
print(f"  collected {len(results['homology'])} homology items")


print("\n[4/4] Multi-task gen...", flush=True)
all_eval = []
with open(EVAL_FILE) as f:
    for line in f: all_eval.append(json.loads(line))
by_cat = defaultdict(list)
for r in all_eval: by_cat[r.get("category", "unknown")].append(r)

random.seed(7)
for cat in ["Cell", "Mol", "Protein", "Literature", "Structure"]:
    if cat not in by_cat: continue
    picks = random.sample(by_cat[cat], min(3, len(by_cat[cat])))
    for r in picks:
        prompt = alpaca(r["instruction"], r.get("input", ""))
        try:
            pred = gen_text(prompt, max_new=200).strip()
        except Exception as e:
            pred = f"[error: {e}]"
        item = {"category": cat,
                "instruction": r["instruction"][:200],
                "input": (r.get("input", "") or "")[:200],
                "reference": r["output"][:400],
                "prediction": pred[:600]}
        results["multi_task"].append(item)
        print(f"  [{cat}] {r['instruction'][:60]}")
        print(f"    pred: {pred[:100]}")
print(f"  collected {len(results['multi_task'])} multi-task items")


# ============== Save MD + JSON ==============
print("\nSaving outputs...", flush=True)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

lines = ["# OmniGene-4-MM Stage 3 v3 — Qualitative Showcase\n",
         f"Model: `{MM_DIR}`\n",
         f"Eval: standard 0.850 / remote 0.695 / struct_recog 1.00 / struct_cap 0.96 / Cell-kw 0.95 / Mol-kw 0.91 / Protein-kw 1.00\n",
         "\n## 1. Vision (Vis-CheBI20)\n"]
for it in results["vision"]:
    lines.append(f"\n### [{it['task']}] {it['image']}\n")
    lines.append(f"**Prompt:** {it['prompt']}\n")
    lines.append(f"**Reference:** {it['reference']}\n")
    lines.append(f"**v3 prediction:** {it['prediction']}\n")

lines.append("\n## 2. Homology (BioPAWS)\n")
for it in results["homology"]:
    lines.append(f"\n### [{it['kind']}] gold label = {it['label']}\n")
    lines.append(f"- Seq1: `{it['seq1_preview']}`\n- Seq2: `{it['seq2_preview']}`\n")
    lines.append(f"**v3 prediction:** {it['prediction']}\n")

lines.append("\n## 3. Multi-task generation\n")
for it in results["multi_task"]:
    lines.append(f"\n### [{it['category']}]\n")
    lines.append(f"**Instruction:** {it['instruction']}\n")
    if it["input"]:
        lines.append(f"**Input:** {it['input']}\n")
    lines.append(f"**Reference:** {it['reference']}\n")
    lines.append(f"**v3 prediction:** {it['prediction']}\n")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"  wrote {OUT_MD}")
print(f"  wrote {OUT_JSON}")
print("Done!")
