"""F4 ESM-2 head baseline: variant-effect prediction the native PLM way.

For ProteinGym DMS, the standard PLM protocol encodes the MUTATED sequence (wild-type with
the point mutation applied) and trains a head to predict effect. We parse the WT sequence
and the mutation (e.g. L199E) from the BioPAWS-2 F4 prompt, apply the mutation, mean-pool
ESM-2 embeddings, and train a binary head (deleterious/benign) on the same train/test split
as the chat models — apples-to-apples on its home turf.

Usage:
  python eval/run_esm_f4.py --task-file data/f4_proteingym_dms.jsonl --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402

ESM = "/root/autodl-tmp/dnagpt/models_local/esm2_3B"
_WT_RE = re.compile(r"Wild-type sequence:\s*([A-Z]+)", re.S)
_MUT_RE = re.compile(r"Mutation:\s*([A-Za-z0-9;,_ ]+)", re.S)
_SINGLE_MUT = re.compile(r"^([A-Z])(\d+)([A-Z])$")


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


def apply_mutations(seq: str, mut_field: str) -> str:
    """Apply one or more point mutations (e.g. 'L199E' or 'A1B:C2D') to seq (1-indexed)."""
    s = list(seq)
    for mut in re.split(r"[;,:]", mut_field.strip()):
        mut = mut.strip()
        m = _SINGLE_MUT.match(mut)
        if not m:
            continue
        wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
        idx = pos - 1
        if 0 <= idx < len(s):
            s[idx] = mt  # apply regardless of WT-match (some assays use offset numbering)
    return "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=1022)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    task = os.path.basename(a.task_file).replace(".jsonl", "")
    train = load_split(a.task_file, "train")
    test = load_split(a.task_file, "test")
    if a.max_train:
        train = train[:a.max_train]
    print(f"[esm-f4] {task}: train={len(train)} test={len(test)}", flush=True)

    def mutated_seqs(rows):
        out = []
        for r in rows:
            u = r["messages"][0]["content"]
            wt = _WT_RE.search(u)
            mut = _MUT_RE.search(u)
            if not wt:
                out.append("A"); continue
            seq = wt.group(1)
            if mut:
                seq = apply_mutations(seq, mut.group(1))
            out.append(seq[:a.max_len])
        return out

    tok = AutoTokenizer.from_pretrained(ESM)
    model = AutoModel.from_pretrained(ESM).cuda().eval()

    def embed(seqs):
        embs = []
        for i in range(0, len(seqs), a.batch_size):
            enc = tok(seqs[i:i + a.batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len).to("cuda")
            with torch.no_grad():
                h = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(pooled.float().cpu())
            if (i // a.batch_size) % 30 == 0:
                print(f"  [embed] {i+a.batch_size}/{len(seqs)}", flush=True)
        return torch.cat(embs, 0)

    choices = train[0].get("choices") or sorted({r["answer_short"] for r in train})
    lab2idx = {c: i for i, c in enumerate(choices)}

    Xtr = embed(mutated_seqs(train))
    Xte = embed(mutated_seqs(test))
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
            loss = lossf(head(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  [head] epoch {ep} loss {tot:.3f}", flush=True)

    head.eval()
    with torch.no_grad():
        pred = head(Xte).argmax(-1).tolist()
    idx2lab = {i: c for c, i in lab2idx.items()}
    preds = {test[i]["id"]: idx2lab[p] for i, p in enumerate(pred)}

    res = score_task(a.task_file, preds)
    res.update({"mode": "sft", "model": "esm2_3B+head", "task": task, "paradigm": "plm_head"})
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"esm2_3B_head__{task}.sft.json")
    json.dump({"result": res, "predictions": preds},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[esm-f4] {res}  -> {out}")


if __name__ == "__main__":
    main()
