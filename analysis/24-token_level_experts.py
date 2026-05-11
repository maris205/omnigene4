#!/usr/bin/env python
# coding: utf-8
"""
24-token_level_experts.py
对每个任务的 top-N "specialty experts", 找出最常路由到该 expert 的 tokens.
回答: 这些 expert 是 motif-specific 还是 distributed?
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _m
_m._can_use_grouped_mm = lambda *a, **k: False

import json
import random
import numpy as np
import torch
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, inject_adapter_in_model

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_V3_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v3-sft-remote"
OUT_DIR = Path("/root/autodl-tmp/dnagpt/outputs/moe_analysis")

NUM_EXPERTS = 128
NUM_LAYERS = 30
SAMPLES_PER_TASK = 50
MAX_LEN = 384

# 来自 23-three_way_compare 的 specialty 排名 (v3): {task: [(layer, expert_id) ...]}
# 我们关注分化最强的 layer 12 + 每个 task 的全局 top-3 expert.
TARGETS = {
    "NL":          [94, 3, 54, 104, 35],
    "Cell":        [100, 6, 97, 30, 86],
    "Mol":         [6, 100, 97, 112, 35],
    "DNA":         [81, 59, 117, 9, 90],
    "Protein":     [57, 78, 9, 51, 83],
    "StdHomology": [9, 57, 107, 83, 103],
    "RemHomology": [114, 48, 72, 115, 110],
    "Structure":   [22, 56, 110, 57, 107],
}
FOCUS_LAYERS = [11, 12, 13, 22, 28]  # 重点观察这些层


def load_task_samples():
    out = {}
    rng = random.Random(42)

    dna_file = Path("/root/autodl-tmp/dnagpt/data/dna_32g.txt")
    if dna_file.exists():
        ls = []
        with dna_file.open() as f:
            for line in f:
                line = line.strip()
                if 100 < len(line) < 800:
                    ls.append(line)
                    if len(ls) >= 200: break
        out["DNA"] = rng.sample(ls, min(SAMPLES_PER_TASK, len(ls)))

    prot_file = Path("/root/autodl-tmp/dnagpt/data/protein_uni_16.txt")
    if prot_file.exists():
        ps = []
        with prot_file.open() as f:
            for line in f:
                line = line.strip()
                if 80 < len(line) < 2000:
                    ps.append(line[:1500])
                    if len(ps) >= 200: break
        out["Protein"] = rng.sample(ps, min(SAMPLES_PER_TASK, len(ps)))

    sds = list(load_dataset('dnagpt/biopaws', 'protein_pair_short')['train'])
    rng.shuffle(sds)
    out["StdHomology"] = [
        f"### Sequence 1:\n{r['sentence1']}\n\n### Sequence 2:\n{r['sentence2']}"
        for r in sds[:SAMPLES_PER_TASK]
    ]

    rds = load_dataset('dnagpt/biopaws', 'protein_pair_remote')['train']
    rds_l = list(rds.select(range(2000)))
    rng.shuffle(rds_l)
    out["RemHomology"] = [
        f"### Sequence 1:\n{r['sentence1']}\n\n### Sequence 2:\n{r['sentence2']}"
        for r in rds_l[:SAMPLES_PER_TASK]
    ]

    owt = Path("/root/autodl-tmp/dnagpt/data/openwebtext.txt")
    if owt.exists():
        nls = []
        with owt.open() as f:
            for line in f:
                line = line.strip()
                if 200 < len(line) < 1500:
                    nls.append(line)
                    if len(nls) >= 200: break
        out["NL"] = rng.sample(nls, min(SAMPLES_PER_TASK, len(nls)))

    cell = Path("/root/autodl-fs/omnigene_v2/sft_data/master/cell_sft_master.jsonl")
    if cell.exists():
        cs = []
        with cell.open() as f:
            for line in f:
                rec = json.loads(line)
                cs.append(f"{rec['instruction']}\n{rec.get('input','')}")
                if len(cs) >= 200: break
        out["Cell"] = rng.sample(cs, min(SAMPLES_PER_TASK, len(cs)))

    mol = Path("/root/autodl-fs/omnigene_v2/sft_data/master/mol_sft_master.jsonl")
    if mol.exists():
        ms = []
        with mol.open() as f:
            for line in f:
                rec = json.loads(line)
                ms.append(f"{rec['instruction']}\n{rec.get('input','')}")
                if len(ms) >= 200: break
        out["Mol"] = rng.sample(ms, min(SAMPLES_PER_TASK, len(ms)))

    st = Path("/root/autodl-tmp/dnagpt/biopaws/sft_data/structure_task_sft.jsonl")
    if st.exists():
        ss = []
        with st.open() as f:
            for line in f:
                rec = json.loads(line)
                msgs = rec.get("messages", [])
                u = ""
                for m in msgs:
                    if m["role"] == "user":
                        u = m["content"]; break
                if u: ss.append(u)
                if len(ss) >= 200: break
        if ss:
            out["Structure"] = rng.sample(ss, min(SAMPLES_PER_TASK, len(ss)))

    return out


def main():
    print("Loading v3 model...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map={"":0})
    lc = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','router.proj'])
    inject_adapter_in_model(lc, model.model.language_model, adapter_name="default")
    ms = model.state_dict()
    sft = torch.load(f"{SFT_V3_DIR}/lora_weights.pt", map_location="cpu")
    n = 0
    for k, v in sft.items():
        if k in ms: ms[k].copy_(v); n += 1
    print(f"  loaded {n} LoRA tensors")
    emb = torch.load(f"{SFT_V3_DIR}/embedding_weights.pt", map_location="cpu")
    model.get_input_embeddings().weight.data.copy_(emb)
    model.eval()
    tok = AutoTokenizer.from_pretrained(SFT_V3_DIR)

    tasks = load_task_samples()
    print("tasks:", {t: len(v) for t, v in tasks.items()})

    # 收集结构: token_records[task][layer][expert] -> Counter[token_str]
    # 同时保留 token id, 因为 BPE 子词显示更直观
    records = {t: {L: {} for L in FOCUS_LAYERS} for t in tasks}

    # 当前 token list 通过闭包传入 hook
    state = {"tokens": None}

    def make_hook(layer_idx):
        def hook(module, inp, output):
            top_k_index = output[2]  # [B*S, K]
            tokens = state["tokens"]  # [B*S] token ids on cpu
            if tokens is None:
                return
            # top_k_index: [B*S, K]
            top_k_index = top_k_index.cpu().numpy()
            for ti in range(top_k_index.shape[0]):
                if ti >= len(tokens): break
                token_id = int(tokens[ti])
                for k in range(top_k_index.shape[1]):
                    expert = int(top_k_index[ti, k])
                    bucket = records  # placeholder
            return
        return hook

    # 简化: 用一个 dict[layer][expert] -> list of (token_id, task)
    # 因为只关心 FOCUS_LAYERS 的全部专家, 不只是 TARGETS, 这样可以一次扫到底
    expert_to_tokens = {(L, e): Counter() for L in FOCUS_LAYERS for e in range(NUM_EXPERTS)}

    def make_hook2(layer_idx, current_task):
        def hook(module, inp, output):
            top_k_index = output[2]  # [B*S, K]
            tokens = state["tokens"]
            if tokens is None or layer_idx not in FOCUS_LAYERS:
                return
            top_k = top_k_index.cpu().numpy()
            B_S, K = top_k.shape
            for ti in range(min(B_S, len(tokens))):
                tid = int(tokens[ti])
                for k in range(K):
                    e = int(top_k[ti, k])
                    expert_to_tokens[(layer_idx, e)][(tid, current_task)] += 1
        return hook

    for task, texts in tasks.items():
        print(f"\n[{task}] hooking...", flush=True)
        # 注册带当前 task 闭包的 hooks
        handles = []
        for i, layer in enumerate(model.model.language_model.layers):
            if i in FOCUS_LAYERS:
                h = layer.router.register_forward_hook(make_hook2(i, task))
                handles.append(h)

        with torch.no_grad():
            for j, txt in enumerate(texts):
                ids = tok(txt, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN).input_ids.to(model.device)
                if ids.shape[1] < 5: continue
                state["tokens"] = ids[0].cpu().numpy()
                model(ids)
                if (j+1) % 10 == 0:
                    print(f"    {j+1}/{len(texts)}", flush=True)
        for h in handles: h.remove()
        state["tokens"] = None

    # 对每个 task 的 target expert, 找该 expert 在 layer 12 上 top tokens
    print("\n=== Top tokens routed to each task's specialty experts (Layer 12) ===")
    report = {}
    for task, top_experts in TARGETS.items():
        report[task] = {}
        for e in top_experts[:3]:  # top-3 expert
            # 整合 layer 11/12/13 三层的 tokens 投票
            combined = Counter()
            for L in [11, 12, 13]:
                combined.update(expert_to_tokens[(L, e)])
            # 该 expert 的总命中
            total = sum(combined.values())
            # 按 task 切分: 是否真的偏好该 task 的 token
            by_task = Counter()
            for (tid, t), c in combined.items():
                by_task[t] += c
            # 分布
            dist_str = ", ".join(f"{t}:{by_task[t]}" for t in tasks)
            print(f"\n{task} -> Expert E{e} (L11-13 combined, total {total} hits)")
            print(f"  task distribution: {dist_str}")

            # 列出 top-15 token (不限 task, 看实际激活分布)
            top_tokens = combined.most_common(15)
            decoded = []
            for (tid, t), c in top_tokens:
                tok_str = tok.decode([tid]).replace("\n", "\\n").replace("\t", "\\t")
                decoded.append((tok_str, t, c))
                print(f"  '{tok_str:<20s}' (task={t:<12s}) x{c}")

            report[task][f"E{e}"] = {
                "total_hits_L11_13": int(total),
                "task_distribution": dict(by_task),
                "top_tokens": [{"token": d[0], "task": d[1], "count": d[2]}
                               for d in decoded],
            }

    with (OUT_DIR / "token_level_experts.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_DIR / 'token_level_experts.json'}")


if __name__ == "__main__":
    main()
