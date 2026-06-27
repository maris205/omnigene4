"""F4 converter: ProteinGym DMS substitutions -> BioPAWS-2 variant-effect QA.

Each ProteinGym assay gives, for a wild-type protein (target_seq from the reference file),
a set of point mutants with a DMS fitness score and a binary bin (1=benign/functional,
0=deleterious). We emit a classification QA per mutant:

  user:      wild-type sequence + the mutation (e.g. H24S) + candidate labels
  assistant: "deleterious" or "benign"

DMS_score is retained in meta.value so the same items can also be scored by Spearman.
This instantiates family F4 (variant effect), the last missing family.

Mutation notation: <WT><pos><MUT>, 1-indexed on target_seq.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import make_sample  # noqa: E402

PG_DIR = "/root/autodl-tmp/dnagpt/biopaws2/data/proteingym"
REF = f"{PG_DIR}/ProteinGym_reference_file_substitutions.csv"


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def build(out_dir: str, per_assay_cap: int):
    import pandas as pd
    ref = pd.read_csv(REF).set_index("DMS_id")
    files = glob.glob(f"{PG_DIR}/ProteinGym_substitutions/*.csv")
    out = os.path.join(out_dir, "f4_proteingym_dms.jsonl")
    n = 0
    from collections import Counter
    sp = Counter(); lab = Counter()
    with open(out, "w", encoding="utf-8") as fh:
        for path in files:
            dms_id = os.path.basename(path).replace(".csv", "")
            if dms_id not in ref.index:
                continue
            seq = str(ref.loc[dms_id, "target_seq"])
            df = pd.read_csv(path)
            if "DMS_score_bin" not in df.columns:
                continue
            # balanced cap per assay
            df = df.sample(frac=1, random_state=0) if len(df) > per_assay_cap else df
            df = df.head(per_assay_cap)
            for _, row in df.iterrows():
                mut = str(row["mutant"])
                score = float(row["DMS_score"]) if "DMS_score" in row else None
                bin_lab = int(row["DMS_score_bin"])
                label = "benign" if bin_lab == 1 else "deleterious"
                # truncate very long sequences to keep prompt manageable
                seq_disp = seq if len(seq) <= 1200 else seq[:1200]
                user = ("Predict the functional effect of the following point mutation(s) on "
                        "this protein. The result will be one of the following: deleterious, "
                        f"benign.\nWild-type sequence: {seq_disp}\nMutation: {mut}")
                split = split_of(f"pg|{dms_id}|{mut}")
                meta = {"value": score} if score is not None else {}
                try:
                    s = make_sample(
                        id=f"proteingym_dms:{n:06d}",
                        task_family="F4_variant", task_id="proteingym_dms",
                        modality=["protein", "text"],
                        messages=[{"role": "user", "content": user},
                                  {"role": "assistant", "content": label}],
                        answer_short=label, choices=["deleterious", "benign"],
                        metric="accuracy", split=split, license="CC-BY-4.0",
                        source=f"ProteinGym:{dms_id}", meta=meta)
                    fh.write(s.to_json() + "\n"); n += 1; sp[split] += 1; lab[label] += 1
                except Exception:
                    pass
    print(f"[F4 proteingym_dms] {n} -> {out} {dict(sp)} labels={dict(lab)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-assay-cap", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build(a.out, a.per_assay_cap)


if __name__ == "__main__":
    main()
