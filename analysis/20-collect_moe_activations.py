#!/usr/bin/env python
# coding: utf-8
"""
20-collect_moe_activations.py
对每个任务采样 N 条文本, 跑 forward, 用 hook 收集每层 router 选中的专家ID.
最终输出: routing_counts[task][layer] -> np.array(num_experts,)  (token-level 计数)

用法:
  python 20-collect_moe_activations.py --tag v3
  python 20-collect_moe_activations.py --tag baseline   # 不加载 LoRA, 只 base
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _m
_m._can_use_grouped_mm = lambda *a, **k: False

import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, inject_adapter_in_model

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_V3_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v3-sft-remote"
CPT_DIR = "/root/autodl-tmp/dnagpt/outputs/gemma-4-26B-A4B-it-bio-cpt-v2"
OUT_DIR = Path("/root/autodl-tmp/dnagpt/outputs/moe_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_EXPERTS = 128
NUM_LAYERS = 30
SAMPLES_PER_TASK = 50
MAX_LEN = 384  # token 数, 每条样本截到这个上限


def load_task_samples():
    """返回 dict[task_name] -> list[str]"""
    out = {}
    rng = random.Random(42)

    # 1. DNA: 直接拿一段
    dna_file = Path("/root/autodl-tmp/dnagpt/data/dna_32g.txt")
    if dna_file.exists():
        dna_lines = []
        with dna_file.open() as f:
            for line in f:
                line = line.strip()
                if 100 < len(line) < 800:
                    dna_lines.append(line)
                    if len(dna_lines) >= 200: break
        out["DNA"] = rng.sample(dna_lines, min(SAMPLES_PER_TASK, len(dna_lines)))

    # 2. Protein
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

    # 3. Standard homology pair
    sds = load_dataset('dnagpt/biopaws', 'protein_pair_short')['train']
    sds = list(sds)
    rng.shuffle(sds)
    pairs = []
    for r in sds[:200]:
        pairs.append(f"### Sequence 1:\n{r['sentence1']}\n\n### Sequence 2:\n{r['sentence2']}")
        if len(pairs) >= SAMPLES_PER_TASK: break
    out["StdHomology"] = pairs

    # 4. Remote homology pair
    rds = load_dataset('dnagpt/biopaws', 'protein_pair_remote')['train']
    rds_l = list(rds.select(range(2000)))
    rng.shuffle(rds_l)
    pairs = []
    for r in rds_l[:200]:
        pairs.append(f"### Sequence 1:\n{r['sentence1']}\n\n### Sequence 2:\n{r['sentence2']}")
        if len(pairs) >= SAMPLES_PER_TASK: break
    out["RemHomology"] = pairs

    # 5. Natural language (OpenWebText sample)
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

    # 6. Cell SFT
    cell = Path("/root/autodl-fs/omnigene_v2/sft_data/master/cell_sft_master.jsonl")
    if cell.exists():
        cs = []
        with cell.open() as f:
            for line in f:
                rec = json.loads(line)
                txt = f"{rec['instruction']}\n{rec.get('input','')}"
                cs.append(txt)
                if len(cs) >= 200: break
        out["Cell"] = rng.sample(cs, min(SAMPLES_PER_TASK, len(cs)))

    # 7. Mol SFT
    mol = Path("/root/autodl-fs/omnigene_v2/sft_data/master/mol_sft_master.jsonl")
    if mol.exists():
        ms = []
        with mol.open() as f:
            for line in f:
                rec = json.loads(line)
                txt = f"{rec['instruction']}\n{rec.get('input','')}"
                ms.append(txt)
                if len(ms) >= 200: break
        out["Mol"] = rng.sample(ms, min(SAMPLES_PER_TASK, len(ms)))

    # 8. Structure (DSSP/3Di) — 从 nl_bio_sft 或 structure_task_sft 拿
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


def load_model(tag):
    """tag in ['v3', 'cpt', 'baseline']."""
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map={"":0})

    if tag in ("v3", "cpt"):
        src = SFT_V3_DIR if tag == "v3" else CPT_DIR
        lc = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','router.proj'])
        inject_adapter_in_model(lc, model.model.language_model, adapter_name="default")
        ms = model.state_dict()
        sft = torch.load(f"{src}/lora_weights.pt", map_location="cpu")
        n = 0
        for k, v in sft.items():
            if k in ms: ms[k].copy_(v); n += 1
        print(f"  loaded {n} LoRA tensors from {src}")
        emb = torch.load(f"{src}/embedding_weights.pt", map_location="cpu")
        model.get_input_embeddings().weight.data.copy_(emb)
        tokenizer = AutoTokenizer.from_pretrained(src)
    else:  # baseline
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model.eval()
    return model, tokenizer


def register_hooks(model, num_layers, num_experts, device):
    """每层 router output -> 累计 expert 选中频次"""
    counts = torch.zeros(num_layers, num_experts, dtype=torch.long, device=device)
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, output):
            # output = (router_probs, top_k_weights, top_k_index)
            # top_k_index: [B*S, K]
            top_k_index = output[2]
            flat = top_k_index.flatten()
            counts[layer_idx].scatter_add_(0, flat,
                torch.ones_like(flat, dtype=torch.long))
        return hook

    for i, layer in enumerate(model.model.language_model.layers):
        h = layer.router.register_forward_hook(make_hook(i))
        handles.append(h)
    return counts, handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", choices=["v3", "cpt", "baseline"], required=True)
    args = parser.parse_args()

    print(f"[tag={args.tag}] Loading model...", flush=True)
    model, tokenizer = load_model(args.tag)

    print("Loading task samples...", flush=True)
    tasks = load_task_samples()
    for t, samples in tasks.items():
        print(f"  {t}: {len(samples)} samples")

    out = {}
    for task, texts in tasks.items():
        print(f"\n[{args.tag}] Task: {task}", flush=True)
        counts, handles = register_hooks(model, NUM_LAYERS, NUM_EXPERTS, model.device)
        total_tokens = 0
        with torch.no_grad():
            for i, txt in enumerate(texts):
                ids = tokenizer(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.to(model.device)
                if ids.shape[1] < 5: continue
                model(ids)
                total_tokens += ids.shape[1]
                if (i+1) % 10 == 0:
                    print(f"    {i+1}/{len(texts)} (tokens so far: {total_tokens})", flush=True)
        for h in handles: h.remove()
        # 每 token 是 top_k_experts 次选择, 归一化到比例
        out[task] = {
            "counts": counts.cpu().numpy(),  # [layers, experts]
            "total_tokens": total_tokens,
            "n_samples": len(texts),
        }

    # 保存
    save_path = OUT_DIR / f"moe_counts_{args.tag}.npz"
    np.savez_compressed(save_path,
        **{f"{t}__counts": v["counts"] for t, v in out.items()},
        **{f"{t}__tokens": np.array([v["total_tokens"]]) for t, v in out.items()})
    print(f"\nSaved {save_path}")

    # 简短摘要
    print("\n=== Summary ===")
    for t, v in out.items():
        c = v["counts"]
        # 每层 top-1 expert
        top_layer_expert = c.argmax(axis=1)  # [layers]
        print(f"  {t}: tokens={v['total_tokens']}, "
              f"top-expert per layer (first 8): {top_layer_expert[:8].tolist()}")


if __name__ == "__main__":
    main()
