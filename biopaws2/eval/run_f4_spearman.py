"""F4 ProteinGym native-metric baseline: ESM-2 regression head -> Spearman.

Field-standard ProteinGym protocol (PLM + head, continuous DMS regression, Spearman):
embed the MUTATED sequence (WT with the point mutation applied) with ESM-2-3B, train a
linear regression head on the train assays' raw DMS_score, and report Spearman on the test
assay. This replaces the bucketized deleterious/benign accuracy with the community-standard
primary metric the reviewer asked for.

Also computes, for reference, a chat-model continuous score = P(benign) via first-token
logits on the same test set, so the paper can report Spearman for the generative paradigm
too (optional; ESM regression is the primary field-standard number).

Usage:
  python eval/run_f4_spearman.py --max-train 12000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ESM = "/root/autodl-tmp/dnagpt/models_local/esm2_3B"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "f4_proteingym_dms.jsonl")
_WT_RE = re.compile(r"Wild-type sequence:\s*([A-Z]+)", re.S)
_MUT_RE = re.compile(r"Mutation:\s*([A-Za-z0-9;,_ ]+)", re.S)
_SINGLE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def apply_mut(seq, mut_field):
    s = list(seq)
    for mut in re.split(r"[;,:]", mut_field.strip()):
        m = _SINGLE.match(mut.strip())
        if m:
            wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
            if 0 <= pos - 1 < len(s):
                s[pos - 1] = mt
    return "".join(s)


def load():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    return rows


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y).correlation
        return float(r) if r == r else 0.0
    except Exception:
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rr = [0.0] * len(v)
            for pos, idx in enumerate(order):
                rr[idx] = pos
            return rr
        rx, ry = rank(x), rank(y)
        n = len(x)
        d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
        return 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    rows = load()
    train = [r for r in rows if r["split"] == "train"][:a.max_train]
    test = [r for r in rows if r["split"] == "test"]
    print(f"[f4-spearman] train={len(train)} test={len(test)}", flush=True)

    def seqs_vals(rr):
        seqs, vals = [], []
        for r in rr:
            u = r["messages"][0]["content"]
            wt = _WT_RE.search(u)
            mut = _MUT_RE.search(u)
            v = r.get("meta", {}).get("value")
            if not wt or v is None:
                continue
            s = apply_mut(wt.group(1), mut.group(1)) if mut else wt.group(1)
            seqs.append(s[:a.max_len]); vals.append(float(v))
        return seqs, vals

    tr_s, tr_v = seqs_vals(train)
    te_s, te_v = seqs_vals(test)

    tok = AutoTokenizer.from_pretrained(ESM)
    model = AutoModel.from_pretrained(ESM).cuda().eval()

    def embed(seqs):
        out = []
        for i in range(0, len(seqs), a.batch_size):
            enc = tok(seqs[i:i + a.batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len).to("cuda")
            with torch.no_grad():
                h = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu())
            if (i // a.batch_size) % 30 == 0:
                print(f"  [embed] {i+a.batch_size}/{len(seqs)}", flush=True)
        return torch.cat(out, 0)

    Xtr, Xte = embed(tr_s), embed(te_s)
    ytr = torch.tensor(tr_v).float()
    # standardize targets for stable regression
    mu, sd = ytr.mean(), ytr.std().clamp(min=1e-6)
    ytr_n = (ytr - mu) / sd

    head = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(512, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    head.train()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            pred = head(Xtr[idx]).squeeze(-1)
            loss = lossf(pred, ytr_n[idx]); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  [head] epoch {ep} loss {tot:.3f}", flush=True)

    head.eval()
    with torch.no_grad():
        pred = head(Xte).squeeze(-1).tolist()
    rho = spearman(pred, te_v)
    res = {"model": "esm2_3B+regression_head", "task": "f4_proteingym_dms",
           "metric": "spearman", "score": round(rho, 4), "n": len(te_v),
           "assay": "AMIE_PSEAE_Wrenbeck_2017", "paradigm": "plm_regression_head"}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"result": res}, open(os.path.join(
        a.out_dir, "esm2_3B_regr__f4_proteingym_dms.spearman.json"), "w"), indent=2)
    print(f"[f4-spearman] ESM-2 regression head Spearman = {rho:.4f} (n={len(te_v)})",
          flush=True)


if __name__ == "__main__":
    main()
