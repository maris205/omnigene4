"""DNABERT-2 head-training baseline for BioPAWS-2 DNA tasks (e.g. promoter detection).

DNABERT-2 (zhihan1996/DNABERT-2-117M) ships custom modeling code written for an older
transformers/torch stack. Two compatibility shims are needed on transformers 5.x / torch 2.8:
  1. config.pad_token_id is missing -> set it.
  2. its Triton flash-attention kernel breaks on torch 2.8 -> force the standard-attention
     path by making `flash_attn_qkvpacked_func` unavailable (the modeling code already has a
     `if flash_attn_qkvpacked_func is None: <standard attention>` fallback).

We patch (2) without touching the model files: after the dynamic module is imported, we set
its `flash_attn_qkvpacked_func` symbol to None.

Mirrors run_plm_head.py: freeze backbone, mean-pool embeddings, train an MLP head on the
same train/test split. Native PLM paradigm — the DNA-model counterpart of ESM-2 for protein.

Usage:
  python eval/run_dnabert_head.py --task-file data/lg_promoter_detection.jsonl --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402

DNABERT = ("/root/autodl-tmp/hf_cache_real/models--zhihan1996--DNABERT-2-117M/"
           "snapshots/7bce263b15377fc15361f52cfab88f8b586abda0")

_DNA_RUN_RE = re.compile(r"([ACGTNacgtn]{20,})")


def parse_dna(user_text: str):
    runs = _DNA_RUN_RE.findall(user_text)
    return max(runs, key=len) if runs else None


def load_split(path, split):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("split") == split:
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, AutoConfig

    cfg = AutoConfig.from_pretrained(DNABERT, trust_remote_code=True)
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = 0
    tok = AutoTokenizer.from_pretrained(DNABERT, trust_remote_code=True)
    model = AutoModel.from_pretrained(DNABERT, config=cfg, trust_remote_code=True)
    # force standard-attention fallback: null out the triton kernel symbol in the module
    mod = sys.modules.get(model.__class__.__module__)
    if mod is not None and hasattr(mod, "flash_attn_qkvpacked_func"):
        mod.flash_attn_qkvpacked_func = None
        print("[dnabert] disabled triton flash-attn -> standard attention", flush=True)
    model = model.cuda().eval()

    task = os.path.basename(a.task_file).replace(".jsonl", "")
    train = load_split(a.task_file, "train")
    test = load_split(a.task_file, "test")
    if a.max_train:
        train = train[:a.max_train]
    print(f"[dnabert-head] {task}: train={len(train)} test={len(test)}", flush=True)
    if not train:
        print("[dnabert-head] no train split, cannot head-train; abort")
        return

    choices = train[0].get("choices") or sorted({r["answer_short"] for r in train})
    lab2idx = {c: i for i, c in enumerate(choices)}

    def embed(rows):
        seqs = [parse_dna(r["messages"][0]["content"]) or "A" for r in rows]
        embs = []
        for i in range(0, len(seqs), a.batch_size):
            chunk = seqs[i:i + a.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_len).to("cuda")
            with torch.no_grad():
                out = model(**enc)
                h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(pooled.float().cpu())
            if (i // a.batch_size) % 30 == 0:
                print(f"  [embed] {i+len(chunk)}/{len(seqs)}", flush=True)
        return torch.cat(embs, 0)

    Xtr = embed(train)
    Xte = embed(test)
    ytr = torch.tensor([lab2idx.get(r["answer_short"], 0) for r in train])

    head = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(512, len(choices)))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    head.train()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(head(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  [head] epoch {ep} loss {tot:.3f}", flush=True)

    head.eval()
    with torch.no_grad():
        pred_idx = head(Xte).argmax(-1).tolist()
    idx2lab = {i: c for c, i in lab2idx.items()}
    preds = {test[i]["id"]: idx2lab[p] for i, p in enumerate(pred_idx)}

    res = score_task(a.task_file, preds)
    res.update({"mode": "sft", "model": "DNABERT-2+head", "task": task,
                "paradigm": "plm_head"})
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"DNABERT-2_head__{task}.sft.json")
    json.dump({"result": res, "predictions": preds},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[dnabert-head] {res}  -> {out}")


if __name__ == "__main__":
    main()
