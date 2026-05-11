#!/usr/bin/env python
# coding: utf-8
"""
15b-eval_v2_sft_remote.py
单独跑 OmniGene-4 v2 (CPT+SFT) 的远缘同源 (protein_pair_remote)
合并到 omnigene4_v2_sft_eval.json
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
SFT_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v2-sft"
RESULT_JSON = "/root/autodl-tmp/dnagpt/biopaws/cpt/omnigene4_v2_sft_eval.json"
SEED = 42

print("Loading OmniGene-4 v2 (CPT+SFT)...", flush=True)
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
    if k in ms:
        ms[k].copy_(v); n += 1
print(f"  Loaded {n} SFT LoRA tensors")
emb = torch.load(f"{SFT_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(emb)
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


print("\n=== Remote Homology (HF: dnagpt/biopaws / protein_pair_remote) ===", flush=True)
ds = load_dataset('dnagpt/biopaws', 'protein_pair_remote')['train']
print(f"  total rows: {len(ds)}")
data0 = [x for x in ds if int(x['label']) == 0]
data1 = [x for x in ds if int(x['label']) == 1]
print(f"  label0: {len(data0)}, label1: {len(data1)}")
random.seed(SEED)
r0 = random.sample(data0, min(1000, len(data0)))
r1 = random.sample(data1, min(1000, len(data1)))
remote_data = r0 + r1
random.shuffle(remote_data)
print(f"  sampled: {len(remote_data)}")

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
print(f"\nRemote homology: {rem_acc:.2%} ({len(rem_true)}/{len(remote_data)}, {rem_failed} unparsed)")

# 合并到已有结果
res = {}
if os.path.exists(RESULT_JSON):
    with open(RESULT_JSON) as f:
        res = json.load(f)
res["remote_homology"] = {
    "accuracy": rem_acc,
    "valid": len(rem_true),
    "total": len(remote_data),
    "failed": rem_failed,
    "source": "dnagpt/biopaws::protein_pair_remote",
}
with open(RESULT_JSON, "w") as f:
    json.dump(res, f, indent=2)
print(f"Updated {RESULT_JSON}")
