"""Integrate BioMatrix-SFT (PPI binding affinity) and protein_catalogue (CoT) -> BioPAWS-2.

  - QizhiPei/BioMatrix-SFT interaction_1d : protein-protein binding affinity (pKd, numeric)
      -> F6_crossmodal / task_id=ppi_affinity. Regression bucketized to low/medium/high
         (edges from data quantiles 5.26 / 6.96); raw value kept in meta.value for Spearman.
  - wanglab/protein_catalogue : sequence -> <think> reasoning </think> functional summary
      -> F7_cot / task_id=protein_catalogue_cot. Free-form (rouge_l).

The BioMatrix input uses special tokens (<|prot_aa_start|><A F><A P>...); we clean them
back to plain residue strings.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import make_sample  # noqa: E402

DATA = "/root/autodl-tmp/dnagpt/omnigene5/data"
PKD_LO, PKD_HI = 5.26, 6.96  # data quantiles (33/66 pct)


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def clean_prot(s: str) -> str:
    s = re.sub(r"<\|?prot_aa_start\|?>|<\|?prot_aa_end\|?>", "", s)
    s = re.sub(r"<A ([A-Z])>", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)  # any remaining angle tokens
    return s.strip()


def bucket(v: float) -> str:
    if v < PKD_LO:
        return "low"
    if v < PKD_HI:
        return "medium"
    return "high"


def conv_biomatrix(out_dir, max_n):
    import pandas as pd
    files = glob.glob(f"{DATA}/H_biomatrix/interaction_1d/*.parquet")
    if not files:
        print("  biomatrix missing, skip"); return
    df = pd.read_parquet(files[0])
    n = 0
    from collections import Counter
    sp = Counter(); lab = Counter()
    out = os.path.join(out_dir, "int_ppi_affinity.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for i, row in enumerate(df.itertuples(index=False)):
            if max_n and n >= max_n:
                break
            try:
                val = float(row.output)
            except (ValueError, TypeError):
                continue
            inp = clean_prot(str(row.input))
            # extract Protein A / B
            m = re.search(r"Protein A:\s*([A-Z]+).*?Protein B:\s*([A-Z]+)", inp, re.S)
            if not m:
                continue
            seqa, seqb = m.group(1), m.group(2)
            b = bucket(val)
            user = ("Predict the protein-protein binding affinity level (pKd) for the two "
                    "proteins. The result will be one of the following: low, medium, high.\n"
                    f"Protein A: {seqa}\nProtein B: {seqb}")
            split = split_of(f"biomatrix|{seqa[:30]}|{seqb[:30]}|{i}")
            try:
                s = make_sample(
                    id=f"ppi_affinity:{i:06d}",
                    task_family="F6_crossmodal", task_id="ppi_affinity",
                    modality=["protein", "text"],
                    messages=[{"role": "user", "content": user},
                              {"role": "assistant", "content": b}],
                    answer_short=b, choices=["low", "medium", "high"],
                    metric="spearman", split=split, license="CC-BY-4.0",
                    source="QizhiPei/BioMatrix-SFT", meta={"value": val})
                fh.write(s.to_json() + "\n"); n += 1; sp[split] += 1; lab[b] += 1
            except Exception:
                pass
    print(f"[int ppi_affinity] {n} -> {out} {dict(sp)} labels={dict(lab)}")


def conv_catalogue(out_dir, max_n):
    import pandas as pd
    files = sorted(glob.glob(f"{DATA}/L_wanglab_protein_catalogue/data/*.parquet"))
    n = 0
    from collections import Counter
    sp = Counter()
    out = os.path.join(out_dir, "int_protein_catalogue_cot.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for path in files:
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            for i, row in enumerate(df.itertuples(index=False)):
                if max_n and n >= max_n:
                    break
                seq = str(getattr(row, "protein", "")).strip()
                gen = str(getattr(row, "generation", "")).strip()
                pid = str(getattr(row, "protein_id", n))
                if not seq or not gen or len(seq) < 10:
                    continue
                user = ("Reason step by step from the protein sequence to its biological "
                        "function, then give a final summary.\nSequence: " + seq)
                split = split_of(f"catalogue|{pid}")
                try:
                    s = make_sample(
                        id=f"protein_catalogue_cot:{n:06d}",
                        task_family="F7_cot", task_id="protein_catalogue_cot",
                        modality=["protein", "text"],
                        messages=[{"role": "user", "content": user},
                                  {"role": "assistant", "content": gen}],
                        answer_short=gen[:200], metric="rouge_l", split=split,
                        license="CC-BY-4.0", source=f"protein_catalogue:{pid}")
                    fh.write(s.to_json() + "\n"); n += 1; sp[split] += 1
                except Exception:
                    pass
            if max_n and n >= max_n:
                break
    print(f"[int protein_catalogue_cot] {n} -> {out} {dict(sp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--catalogue-max", type=int, default=30000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conv_biomatrix(a.out, None)
    conv_catalogue(a.out, a.catalogue_max)


if __name__ == "__main__":
    main()
