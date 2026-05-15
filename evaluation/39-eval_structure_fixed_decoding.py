#!/usr/bin/env python
# coding: utf-8
"""
39-eval_structure_fixed_decoding.py

针对 Structure/Mutation 失败的快速修复尝试:
  1. 调整 prompt template (去掉 <Assistant> chat tag)
  2. 用 sampling decoding (temp=0.7, top_p=0.9, repetition_penalty=1.2)
  3. 阻止生成 <User>/<Assistant> tokens

测试 BF16 完整模型，不重训。
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
OUT_JSON = "/root/autodl-tmp/dnagpt/outputs/structure_fixed_eval.json"

N_SAMPLES = 50
MAX_NEW_TOKENS = 250
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

print("Loading BF16 v3...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()

# 找出 <Assistant> 和 <User> 的 token id, 用来 ban
bad_tokens = []
for tag in ["<Assistant>", "<User>", "Assistant", "User"]:
    enc = tokenizer.encode(tag, add_special_tokens=False)
    bad_tokens.extend(enc)
bad_tokens = list(set(bad_tokens))
print(f"  Banned token IDs: {bad_tokens}")


# 三种 prompt 模板做对比
def template_A_chat(instr, inp):
    """原始 chat template (会触发坍缩)"""
    if inp.strip():
        return f"<User>\n### Instruction:\n{instr}\n\n{inp}\n### Answer:\n<Assistant>\n"
    return f"<User>\n### Instruction:\n{instr}\n\n### Answer:\n<Assistant>\n"

def template_B_alpaca(instr, inp):
    """纯 Alpaca 风格, 不带 chat tag"""
    if inp.strip():
        return f"### Instruction:\n{instr}\n\n{inp}\n\n### Answer:\n"
    return f"### Instruction:\n{instr}\n\n### Answer:\n"

def template_C_bare(instr, inp):
    """极简, 直接给上下文"""
    if inp.strip():
        return f"{instr}\n\n{inp}\n\nAnswer: "
    return f"{instr}\n\nAnswer: "


def generate(prompt, do_sample=False, ban_tokens=False, max_tokens=MAX_NEW_TOKENS):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    kw = dict(
        max_new_tokens=max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        kw.update(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2)
    else:
        kw.update(do_sample=False)
    if ban_tokens:
        kw["bad_words_ids"] = [[t] for t in bad_tokens]

    with torch.no_grad():
        out = model.generate(ids, **kw)
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def collapse_score(text):
    t = re.sub(r'[^A-Z]', '', text.upper())
    if len(t) < 5:
        return 0.0
    return len(set(t)) / 20.0


def char_overlap(pred, ref):
    p = re.sub(r'[^A-Z]', '', pred.upper())
    r = re.sub(r'[^A-Z]', '', ref.upper())
    if not p or not r:
        return 0.0
    L = min(len(p), len(r))
    if L == 0:
        return 0.0
    return sum(1 for i in range(L) if p[i] == r[i]) / L


# 读 Structure
all_data = []
with open(EVAL_FILE) as f:
    for line in f:
        r = json.loads(line)
        if r.get("category") == "Structure":
            all_data.append(r)
samples = random.sample(all_data, N_SAMPLES)
print(f"Testing on {len(samples)} Structure samples")

configs = {
    "A_chat_greedy": {"template": template_A_chat, "do_sample": False, "ban_tokens": False},
    "B_alpaca_greedy": {"template": template_B_alpaca, "do_sample": False, "ban_tokens": False},
    "C_alpaca_sampling": {"template": template_B_alpaca, "do_sample": True, "ban_tokens": False},
    "D_alpaca_ban_tags": {"template": template_B_alpaca, "do_sample": False, "ban_tokens": True},
    "E_bare_greedy": {"template": template_C_bare, "do_sample": False, "ban_tokens": False},
}

all_results = {}
for cfg_name, cfg in configs.items():
    print(f"\n=== Config: {cfg_name} ===", flush=True)
    overlaps = []
    divs = []
    collapse = 0
    asst_repeats = 0
    samples_out = []
    for i, item in enumerate(samples):
        if i % 10 == 0:
            print(f"  {i}/{len(samples)}", flush=True)
        prompt = cfg["template"](item["instruction"], item.get("input", ""))
        try:
            pred = generate(prompt, do_sample=cfg["do_sample"], ban_tokens=cfg["ban_tokens"]).strip()
        except Exception as e:
            print(f"    error: {e}")
            continue

        ref = item["output"].strip()
        ov = char_overlap(pred, ref)
        dv = collapse_score(pred)
        if dv < 0.15:
            collapse += 1
        if pred.count("Assistant") > 2:
            asst_repeats += 1
        overlaps.append(ov)
        divs.append(dv)
        if i < 3:
            samples_out.append({"ref": ref[:120], "pred": pred[:150], "overlap": ov})

    avg_ov = sum(overlaps) / len(overlaps) if overlaps else 0
    avg_dv = sum(divs) / len(divs) if divs else 0
    print(f"  Overlap: {avg_ov:.3f}, Diversity: {avg_dv:.3f}, Collapse: {collapse}/{len(overlaps)}, Asst-repeat: {asst_repeats}/{len(overlaps)}")
    all_results[cfg_name] = {
        "avg_char_overlap": avg_ov,
        "avg_diversity": avg_dv,
        "collapse_count": collapse,
        "assistant_repeat_count": asst_repeats,
        "n": len(overlaps),
        "samples": samples_out,
    }

print("\n=== Summary ===")
for cfg_name, r in all_results.items():
    print(f"  {cfg_name:<25s}: overlap={r['avg_char_overlap']:.3f} diversity={r['avg_diversity']:.3f} collapse={r['collapse_count']}/{r['n']} asst-repeat={r['assistant_repeat_count']}/{r['n']}")

with open(OUT_JSON, "w") as f:
    json.dump({
        "model": "OmniGene-4-SFT-v3 BF16",
        "task": "Structure (3Di/DSSP) — decoding strategy comparison",
        "n_samples": N_SAMPLES,
        "configs": all_results,
    }, f, indent=2)
print(f"\nSaved {OUT_JSON}")
