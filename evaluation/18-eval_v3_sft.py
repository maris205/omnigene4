#!/usr/bin/env python
# coding: utf-8
"""
18-eval_v3_sft.py
评测 OmniGene-4 v3 (CPT + Bio-SFT v3, remote-augmented)
跑 Standard / Remote / BixBench, 与 v2 对比。
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import torch, random, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, inject_adapter_in_model
from sklearn.metrics import accuracy_score

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v3-sft-remote"
RESULT_JSON = "/root/autodl-tmp/dnagpt/biopaws/cpt/omnigene4_v3_sft_eval.json"
SEED = 42

print("Loading OmniGene-4 v3 (CPT + Bio-SFT v3 remote)...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map={"":0})
lc = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','router.proj'])
inject_adapter_in_model(lc, model.model.language_model, adapter_name="default")
ms = model.state_dict()
sft = torch.load(f"{SFT_DIR}/lora_weights.pt", map_location="cpu")
n = 0
for k, v in sft.items():
    if k in ms: ms[k].copy_(v); n += 1
print(f"  Loaded {n} SFT LoRA tensors")
emb = torch.load(f"{SFT_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(emb)
print(f"  Loaded SFT embedding")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(SFT_DIR)


def generate(prompt, max_tokens=8):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def parse_homology(text):
    t = text.strip().lower()
    head = t[:40]
    if 'non-homolog' in head or 'non homolog' in head or 'nonhomolog' in head:
        return 0
    if 'homolog' in head:
        return 1
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


# ========== Standard ==========
print("\n=== Standard Homology ===", flush=True)
ds = load_dataset('dnagpt/biopaws', 'protein_pair_short')['train']
data0 = [x for x in ds if x['label'] == 0]
data1 = [x for x in ds if x['label'] == 1]
random.seed(SEED)
s0 = random.sample(data0, int(len(data0) * 0.3))
s1 = random.sample(data1, int(len(data1) * 0.3))
std_data = s0 + s1
random.shuffle(std_data)

st_t, st_p, st_f = [], [], 0
for i, item in enumerate(std_data):
    if i % 500 == 0:
        print(f"  Standard {i}/{len(std_data)}", flush=True)
    resp = generate(make_prompt(item['sentence1'], item['sentence2']))
    pred = parse_homology(resp)
    if pred is not None:
        st_t.append(item['label']); st_p.append(pred)
    else:
        st_f += 1
std_acc = accuracy_score(st_t, st_p) if st_t else 0.0
print(f"Standard: {std_acc:.2%} ({len(st_t)}/{len(std_data)}, {st_f} unparsed)")

# ========== Remote ==========
print("\n=== Remote Homology ===", flush=True)
rds = load_dataset('dnagpt/biopaws', 'protein_pair_remote')['train']
data0 = [x for x in rds if int(x['label']) == 0]
data1 = [x for x in rds if int(x['label']) == 1]
random.seed(SEED)
r0 = random.sample(data0, min(1000, len(data0)))
r1 = random.sample(data1, min(1000, len(data1)))
rem_data = r0 + r1
random.shuffle(rem_data)

rt, rp, rf = [], [], 0
for i, item in enumerate(rem_data):
    if i % 200 == 0:
        print(f"  Remote {i}/{len(rem_data)}", flush=True)
    resp = generate(make_prompt(item['sentence1'], item['sentence2']))
    pred = parse_homology(resp)
    if pred is not None:
        rt.append(int(item['label'])); rp.append(pred)
    else:
        rf += 1
rem_acc = accuracy_score(rt, rp) if rt else 0.0
print(f"Remote: {rem_acc:.2%} ({len(rt)}/{len(rem_data)}, {rf} unparsed)")

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
    p = (
        "<User>\n### Instruction:\n"
        "Based on the research result below, determine if the hypothesis is True or False.\n"
        "Answer only True or False.\n\n"
        f"### Hypothesis:\n{hypothesis}\n\n"
        f"### Research Result:\n{result[:500]}\n\n"
        "### Answer:\n<Assistant>\n"
    )
    resp = generate(p, 5).strip().lower()
    pred = 'True' if 'true' in resp[:20] else 'False' if 'false' in resp[:20] else None
    if pred:
        total += 1
        if pred == answer:
            correct += 1
    if total % 30 == 0 and total > 0:
        print(f"  BixBench {total} done, acc={correct/total:.2%}", flush=True)
bix_acc = correct / total if total else 0
print(f"BixBench: {correct}/{total} = {bix_acc:.2%}")

results = {
    "standard_homology": {"accuracy": std_acc, "valid": len(st_t), "total": len(std_data), "failed": st_f},
    "remote_homology": {"accuracy": rem_acc, "valid": len(rt), "total": len(rem_data), "failed": rf,
                        "source": "dnagpt/biopaws::protein_pair_remote"},
    "bixbench_knowledge": {"accuracy": bix_acc, "correct": correct, "total": total},
}
with open(RESULT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {RESULT_JSON}")
