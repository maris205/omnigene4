"""BioPAWS-2 PLM baseline: native-paradigm head-training (Mode B for encoder PLMs).

Fairness principle ("dual-track, each on home turf"): specialized protein/DNA language
models (ESM-2, ProtT5, DNABERT-2) are NOT asked to answer free-form QA — that is not their
native interface and would understate them. Instead they run their *standard* protocol:

  - freeze (or optionally unfreeze) the backbone,
  - extract a pooled embedding per sequence,
  - train a small classification / regression head on the SAME BioPAWS-2 train split,
  - evaluate on the SAME held-out test split.

This is the apples-to-apples counterpart of LLM QA-LoRA-SFT: identical data, identical
splits, each model in the training regime it was designed for.

Supported task shapes (read from the QA jsonl, sequences recovered from the user turn):
  - pairwise (F1): siamese encode seq1/seq2 -> [|u-v|, u*v, u, v] -> binary head
  - single-seq classification (F2/F3/F5): encode seq -> linear head over `choices`
  - single-seq regression->bucket (F2/F4): encode seq -> regression head; bucket for acc,
    raw for spearman.

The sequence is parsed back out of the instruction text (we wrote it, so the markers are
known: 'Sequence 1: ... Sequence 2: ...', 'Sequence: ...', '>seqN').

Usage:
  python eval/run_plm_head.py --task-file data/protein_homology_std.jsonl \
      --plm /root/autodl-tmp/dnagpt/models_local/esm2_3B --shape pairwise --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402


# ---- sequence recovery from our own templates --------------------------------

_SEQ_PAIR_RE = [
    re.compile(r"Sequence 1:\s*(\S+).*?Sequence 2:\s*(\S+)", re.S),
    re.compile(r"Protein A:\s*(\S+).*?Protein B:\s*(\S+)", re.S),
    re.compile(r">seq1\s*(\S+)\s*>seq2\s*(\S+)", re.S),
    re.compile(r"A:\s*(\S+).*?B:\s*(\S+)", re.S),
    re.compile(r"CDS:\s*(\S+).*?Protein:\s*(\S+)", re.S),
    re.compile(r"DNA:\s*(\S+).*?Protein:\s*(\S+)", re.S),
    re.compile(r">dna1\s*(\S+)\s*>dna2\s*(\S+)", re.S),
]
_SEQ_SINGLE_RE = [
    re.compile(r"Sequence:\s*(\S+)", re.S),
    re.compile(r"sequence[:\s]+(\S+)", re.S | re.I),
]
# fallback: longest residue/nucleotide-like run (llama-gene appends the raw sequence on its
# own line after the instruction, with no "Sequence:" marker)
_SEQ_RUN_RE = re.compile(r"([ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]{20,})")


def parse_pair(user_text: str):
    for rx in _SEQ_PAIR_RE:
        m = rx.search(user_text)
        if m:
            return m.group(1), m.group(2)
    return None, None


def parse_single(user_text: str):
    for rx in _SEQ_SINGLE_RE:
        m = rx.search(user_text)
        if m and len(m.group(1)) >= 10:
            return m.group(1)
    runs = _SEQ_RUN_RE.findall(user_text)
    if runs:
        return max(runs, key=len)
    return None


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


def embed_sequences(plm_path, seqs, batch_size=16, max_len=1024, device="cuda"):
    """Mean-pooled embeddings from an ESM-style encoder. ProtT5/DNABERT handled by
    model_type branch."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(plm_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(plm_path, trust_remote_code=True).to(device).eval()
    embs = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        # ESM expects space-free; ProtT5 expects spaced residues
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        with torch.no_grad():
            out = model(**enc)
            h = out.last_hidden_state            # [B, L, H]
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(pooled.float().cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  [embed] {i+len(chunk)}/{len(seqs)}", flush=True)
    return torch.cat(embs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--plm", required=True)
    ap.add_argument("--shape", choices=["pairwise", "single_cls", "single_reg"],
                    default="pairwise")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=None)
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    task = os.path.basename(a.task_file).replace(".jsonl", "")
    train = load_split(a.task_file, "train")
    test = load_split(a.task_file, "test")
    if a.max_train:
        train = train[:a.max_train]
    print(f"[plm-head] {task}: train={len(train)} test={len(test)} plm={a.plm} shape={a.shape}")

    def user_of(r):
        return r["messages"][0]["content"]

    # --- recover sequences + labels ---
    def build_xy(rows):
        seqs1, seqs2, labels, raw = [], [], [], []
        for r in rows:
            if a.shape == "pairwise":
                s1, s2 = parse_pair(user_of(r))
                if not s1 or not s2:
                    continue
                seqs1.append(s1); seqs2.append(s2)
            else:
                s = parse_single(user_of(r))
                if not s:
                    continue
                seqs1.append(s)
            labels.append(r["answer_short"])
            raw.append(r)
        return seqs1, seqs2, labels, raw

    tr1, tr2, trl, _ = build_xy(train)
    te1, te2, tel, te_raw = build_xy(test)

    # label vocab (classification) from train choices
    choices = train[0].get("choices") or sorted(set(trl))
    lab2idx = {c: i for i, c in enumerate(choices)}

    # --- embed ---
    if a.shape == "pairwise":
        e_tr1 = embed_sequences(a.plm, tr1, a.batch_size)
        e_tr2 = embed_sequences(a.plm, tr2, a.batch_size)
        e_te1 = embed_sequences(a.plm, te1, a.batch_size)
        e_te2 = embed_sequences(a.plm, te2, a.batch_size)
        def feat(u, v):
            return torch.cat([torch.abs(u - v), u * v, u, v], -1)
        Xtr = feat(e_tr1, e_tr2); Xte = feat(e_te1, e_te2)
    else:
        Xtr = embed_sequences(a.plm, tr1, a.batch_size)
        Xte = embed_sequences(a.plm, te1, a.batch_size)

    ytr = torch.tensor([lab2idx.get(l, 0) for l in trl])
    in_dim = Xtr.shape[1]
    n_cls = len(choices)
    head = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(512, n_cls))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    head.train()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            logits = head(Xtr[idx])
            loss = lossf(logits, ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  [head] epoch {ep} loss {tot:.3f}", flush=True)

    head.eval()
    with torch.no_grad():
        pred_idx = head(Xte).argmax(-1).tolist()
    idx2lab = {i: c for c, i in lab2idx.items()}
    preds = {te_raw[i]["id"]: idx2lab[p] for i, p in enumerate(pred_idx)}

    res = score_task(a.task_file, preds)
    res.update({"mode": "sft", "model": f"{os.path.basename(a.plm)}+head", "task": task,
                "paradigm": "plm_head"})
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{os.path.basename(a.plm)}_head"
    out = os.path.join(a.out_dir, f"{tag}__{task}.sft.json")
    json.dump({"result": res, "predictions": preds},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[plm-head] {res}  -> {out}")


if __name__ == "__main__":
    main()
