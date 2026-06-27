"""F1 converter: protein homology (standard + remote) -> instruction-tuning QA jsonl.

Reads the local `dnagpt/biopaws` parquet (sentence1, sentence2, label) and emits
schema-valid `messages` records, cycling instruction templates by index.

Split strategy: deterministic hash-based split on the pair so that train/val/test are
disjoint and reproducible (no RNG — see templates.py note). For a leakage-clean public
release we additionally run scripts/leakage_check.py (MMseqs2 clustering); this converter's
hash split is the fast default used during development.

Usage:
    python -m tasks.f1_pairwise.protein_homology --task protein_homology_std --out data/
    python -m tasks.f1_pairwise.protein_homology --task protein_homology_remote \
        --out data/ --max 40000
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

# allow `python tasks/f1_pairwise/protein_homology.py` and `-m` both
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from schema import make_sample  # noqa: E402
from tasks import templates  # noqa: E402

PARQUET_BASE = ("/root/autodl-tmp/hf_cache_real/hub/datasets--dnagpt--biopaws/"
                "snapshots/6404c36e42e9f7b83cadcdd70029042757eb8b57")

TASK_TO_PARQUET = {
    "protein_homology_std": f"{PARQUET_BASE}/protein_pair_short/train-00000-of-00001.parquet",
    "protein_homology_remote": f"{PARQUET_BASE}/protein_pair_remote/train-00000-of-00001.parquet",
}
TASK_TO_FAMILY = {
    "protein_homology_std": ("F1_pairwise", "dnagpt/biopaws:protein_pair_short"),
    "protein_homology_remote": ("F1_pairwise", "dnagpt/biopaws:protein_pair_remote"),
}
# template key in templates.template_for()
TASK_TO_TPLKEY = {
    "protein_homology_std": "protein_homology_std",
    "protein_homology_remote": "protein_homology_remote",
}


def split_of(key: str) -> str:
    """Deterministic 80/10/10 split from a stable hash."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def build(task: str, out_dir: str, max_n: int | None = None, explain: bool = False) -> str:
    import pandas as pd

    parquet = TASK_TO_PARQUET[task]
    family, source = TASK_TO_FAMILY[task]
    tpls = templates.template_for(TASK_TO_TPLKEY[task])

    df = pd.read_parquet(parquet)
    if max_n is not None and len(df) > max_n:
        # balanced subsample: keep first max_n/2 of each label, deterministic
        half = max_n // 2
        pos = df[df["label"] == 1].head(half)
        neg = df[df["label"] == 0].head(half)
        df = pd.concat([pos, neg]).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}.jsonl")
    counts = {"train": 0, "val": 0, "test": 0}

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(df.itertuples(index=False)):
            seq1, seq2, label = row.sentence1, row.sentence2, int(row.label)
            # normalize residue casing: the remote split is stored lowercase while the
            # standard split is uppercase; uppercasing makes the QA surface form uniform
            # (the model was SFT'd on uppercase residues, so lowercase caused mode collapse).
            seq1, seq2 = seq1.upper(), seq2.upper()
            tpl = tpls[i % len(tpls)]
            user = tpl.format(seq1=seq1, seq2=seq2)
            ans = templates.YESNO[label]
            assistant = ans + (templates.HOMOLOGY_EXPLAIN[label] if explain else "")
            uid = f"{task}:{i:06d}"
            split = split_of(f"{task}|{seq1[:40]}|{seq2[:40]}|{label}")
            s = make_sample(
                id=uid,
                task_family=family,
                task_id=task,
                modality=["protein", "text"],
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                answer_short=ans,
                choices=["Yes", "No"],
                metric="accuracy",
                split=split,
                license="CC-BY-4.0",
                source=source,
            )
            fh.write(s.to_json() + "\n")
            counts[split] += 1

    print(f"[F1 {task}] wrote {sum(counts.values())} samples -> {out_path}")
    print(f"           splits: {counts}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_TO_PARQUET))
    ap.add_argument("--out", default="data")
    ap.add_argument("--max", type=int, default=None, help="balanced subsample cap")
    ap.add_argument("--explain", action="store_true", help="append CoT-style explanation")
    a = ap.parse_args()
    build(a.task, a.out, a.max, a.explain)


if __name__ == "__main__":
    main()
