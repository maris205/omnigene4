#!/usr/bin/env python
# coding: utf-8
"""
15-eval_v2_sft.py
评测 OmniGene-4 v2 (CPT 0.6 + Bio-SFT v1)
- Standard homology
- Remote homology
- BixBench Knowledge
- 对比: v1 / v2(0.6 CPT only) / v2(CPT+SFT) / Gemma-4 Instruct
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import torch
import random
import re
import json
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, inject_adapter_in_model
from sklearn.metrics import accuracy_score, classification_report

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
CPT_DIR = "/root/autodl-tmp/dnagpt/outputs/gemma-4-26B-A4B-it-bio-cpt-v2"
SFT_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v2-sft"
REMOTE_CSV = "/root/autodl-fs/omnigene_v2/data/protein_pair_remote.csv"
SEED = 42

print("Loading OmniGene-4 v2 (CPT + Bio-SFT)...", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map={"": 0},
)
lora_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
inject_adapter_in_model(lora_config, model.model.language_model, adapter_name="default")
model_state = model.state_dict()

# 加载 SFT LoRA (已经包含 CPT 基础上的进一步更新)
sft_lora = torch.load(f"{SFT_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
for k, v in sft_lora.items():
    if k in model_state:
        model_state[k].copy_(v); loaded += 1
print(f"  Loaded {loaded} SFT LoRA tensors")

# 加载 SFT 阶段的 embedding (包含 CPT + SFT 所有更新)
sft_embed = torch.load(f"{SFT_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(sft_embed)
print(f"  Loaded SFT embedding weights")

model.eval()
tokenizer = AutoTokenizer.from_pretrained(SFT_DIR)


def generate(prompt, max_tokens=8):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def parse_homology(text):
    """SFT v2 输出 'Homologous' / 'Non-Homologous'. 先判 non."""
    t = text.strip().lower()
    head = t[:40]
    if 'non-homolog' in head or 'non homolog' in head or 'nonhomolog' in head:
        return 0
    if 'homolog' in head:
        return 1
    # 兼容 yes/no
    if 'yes' in head: return 1
    if 'no' in head: return 0
    return None


def make_prompt(seq1, seq2):
    return (
        "<User>\n### Instruction:\n"
        "Determine if the two sequences below are structurally related (like paraphrases).\n\n"
        f"### Sequence 1:\n{seq1}\n\n"
        f"### Sequence 2:\n{seq2}\n\n"
        "### Answer:\n<Assistant>\n"
    )


# ========== Standard Homology ==========
print("\n=== Standard Homology ===", flush=True)
ds = load_dataset('dnagpt/biopaws', 'protein_pair_short')['train']
data0 = [x for x in ds if x['label'] == 0]
data1 = [x for x in ds if x['label'] == 1]
random.seed(SEED)
s0 = random.sample(data0, int(len(data0) * 0.3))
s1 = random.sample(data1, int(len(data1) * 0.3))
std_data = s0 + s1
random.shuffle(std_data)

std_true, std_pred, std_failed = [], [], 0
for i, item in enumerate(std_data):
    if i % 300 == 0:
        print(f"  Standard {i}/{len(std_data)}", flush=True)
    resp = generate(make_prompt(item['sentence1'], item['sentence2']), 8)
    pred = parse_homology(resp)
    if pred is not None:
        std_true.append(item['label']); std_pred.append(pred)
    else:
        std_failed += 1
std_acc = accuracy_score(std_true, std_pred) if std_true else 0.0
print(f"Standard homology: {std_acc:.2%} ({len(std_true)}/{len(std_data)})")

# ========== Remote Homology ==========
print("\n=== Remote Homology ===", flush=True)
if os.path.exists(REMOTE_CSV):
    df = pd.read_csv(REMOTE_CSV)
    rem = df.to_dict('records')
    data0 = [x for x in rem if int(x['label']) == 0]
    data1 = [x for x in rem if int(x['label']) == 1]
    random.seed(SEED)
    r0 = random.sample(data0, min(1000, len(data0)))
    r1 = random.sample(data1, min(1000, len(data1)))
    remote_data = r0 + r1
    random.shuffle(remote_data)

    rem_true, rem_pred, rem_failed = [], [], 0
    for i, item in enumerate(remote_data):
        if i % 200 == 0:
            print(f"  Remote {i}/{len(remote_data)}", flush=True)
        resp = generate(make_prompt(item['sentence1'], item['sentence2']), 8)
        pred = parse_homology(resp)
        if pred is not None:
            rem_true.append(int(item['label'])); rem_pred.append(pred)
        else:
            rem_failed += 1
    rem_acc = accuracy_score(rem_true, rem_pred) if rem_true else 0.0
    print(f"Remote homology: {rem_acc:.2%} ({len(rem_true)}/{len(remote_data)})")
else:
    rem_acc, rem_true, rem_failed = None, [], 0
    print(f"Skip remote: {REMOTE_CSV} not found")

# ========== BixBench ==========
print("\n=== BixBench Knowledge ===", flush=True)
bix = load_dataset('futurehouse/BixBench', split='train')
correct, total = 0, 0
for item in bix:
    answer = str(item.get('answer', '')).strip()
    if answer not in ['True', 'False']:
        continue
    hypothesis = item.get('hypothesis', '')
    result = item.get('result', '')
    if not hypothesis or not result:
        continue
    prompt = (
        "<User>\n### Instruction:\n"
        "Based on the research result below, determine if the hypothesis is True or False.\n"
        "Answer only True or False.\n\n"
        f"### Hypothesis:\n{hypothesis}\n\n"
        f"### Research Result:\n{result[:500]}\n\n"
        "### Answer:\n<Assistant>\n"
    )
    resp = generate(prompt, 5).strip().lower()
    pred = 'True' if 'true' in resp[:20] else 'False' if 'false' in resp[:20] else None
    if pred:
        total += 1
        if pred == answer:
            correct += 1
    if total % 30 == 0 and total > 0:
        print(f"  BixBench {total} done, acc={correct/total:.2%}", flush=True)
bix_acc = correct / total if total else 0
print(f"BixBench Knowledge: {correct}/{total} = {bix_acc:.2%}")

results = {
    "standard_homology": {"accuracy": std_acc, "valid": len(std_true), "failed": std_failed},
    "remote_homology": {"accuracy": rem_acc, "valid": len(rem_true)} if rem_acc is not None else None,
    "bixbench_knowledge": {"accuracy": bix_acc, "correct": correct, "total": total},
}
with open("omnigene4_v2_sft_eval.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to omnigene4_v2_sft_eval.json")
