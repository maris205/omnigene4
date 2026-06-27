"""F7 converter: bioreason (wanglab/bioreason-pro-sft-reasoning-data) -> BioPAWS-2 CoT QA.

The bioreason corpus pairs a protein sequence (+ InterPro/GO annotation context) with a
step-by-step `reasoning` trace and a `final_answer` functional summary. This is exactly the
'mental-folding' style chain-of-thought BioPAWS-1 demonstrated qualitatively (reasoning from
domain architecture to function), now at scale. We emit it as F7 free-form CoT QA:

  user:      "Reason step by step from the sequence/domain evidence to the protein's
              function." + sequence
  assistant: <reasoning>  ...  <final answer>

Scored by ROUGE-L against the reference reasoning+answer (free-form). Contributes the
biological-reasoning tier of BioPAWS-2.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import make_sample  # noqa: E402

BIOREASON = "/root/autodl-tmp/dnagpt/biopaws2/data/bioreason/validation-00000-of-00001.parquet"


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def build(out_dir: str, max_n: int | None = None):
    import pandas as pd
    df = pd.read_parquet(BIOREASON)
    df = df[df["reasoning"].notna() & df["sequence"].notna()]
    if max_n:
        df = df.head(max_n)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "f7_bioreason_cot.jsonl")

    n = 0
    from collections import Counter
    sp = Counter()
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(df.itertuples(index=False)):
            seq = str(r.sequence).strip()
            reasoning = str(r.reasoning).strip()
            final = str(r.final_answer).strip() if r.final_answer is not None else ""
            if not seq or not reasoning:
                continue
            # optional annotation context (InterPro / organism) to ground the reasoning
            ctx_bits = []
            if getattr(r, "organism", None):
                ctx_bits.append(f"Organism: {str(r.organism).strip()}")
            if getattr(r, "interpro_formatted", None):
                ip = str(r.interpro_formatted).strip()
                if ip and ip.lower() != "nan":
                    ctx_bits.append(f"InterPro domains:\n{ip[:600]}")
            ctx = ("\n".join(ctx_bits) + "\n") if ctx_bits else ""
            user = (
                "Reason step by step from the protein sequence and its domain evidence to "
                "the protein's biological function (a 'mental-folding' analysis). Provide "
                "your reasoning, then a final functional summary.\n"
                f"{ctx}Sequence: {seq}")
            assistant = reasoning + (("\n\nFinal answer: " + final) if final else "")
            split = split_of(f"bioreason|{seq[:40]}|{i}")
            try:
                s = make_sample(
                    id=f"bioreason_cot:{i:05d}",
                    task_family="F7_cot", task_id="bioreason_cot",
                    modality=["protein", "text"],
                    messages=[{"role": "user", "content": user},
                              {"role": "assistant", "content": assistant}],
                    answer_short=(final[:200] if final else reasoning[:200]),
                    metric="rouge_l", split=split,
                    license="CC-BY-4.0", source="wanglab/bioreason-pro-sft-reasoning-data")
                fh.write(s.to_json() + "\n")
                n += 1
                sp[split] += 1
            except Exception:
                pass
    print(f"[F7 bioreason_cot] wrote {n} -> {out_path}  splits={dict(sp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    build(a.out, a.max)


if __name__ == "__main__":
    main()
