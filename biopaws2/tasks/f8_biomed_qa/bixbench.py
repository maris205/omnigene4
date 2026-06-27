"""F8 converter: BixBench -> BioPAWS-2 biomedical sequence/data-grounded QA.

BixBench (futurehouse/BixBench) is a bioinformatics reasoning benchmark where each item is
grounded in real RNA-seq / genomics / data-analysis tasks (not pure literature recall). Two
QA forms are emitted, both in the BioPAWS-2 candidate-label style:

  - MCQ:  question + {ideal, distractors} -> multiple-choice (shuffled), answer = ideal.
  - T/F:  hypothesis + result -> True/False (the form the main paper already uses).

These are held-out evaluation items (no train split in BixBench); we mark all as test, so
F8 contributes to zero-shot / generalist evaluation but not to SFT training. This is the
'general biological reasoning' tier of BioPAWS-2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import make_sample  # noqa: E402

BIX_GLOB = ("/root/autodl-tmp/hf_cache_real/hub/datasets--futurehouse--BixBench/"
            "snapshots/*/BixBench.jsonl")


def stable_shuffle(options, key):
    """Deterministic option ordering from a hash (no RNG)."""
    h = hashlib.md5(key.encode()).hexdigest()
    return [o for _, o in sorted(zip(
        [int(h[i % len(h):i % len(h) + 4], 16) for i in range(len(options))], options))]


def load_bix():
    import glob
    path = glob.glob(BIX_GLOB)
    if not path:
        raise FileNotFoundError("BixBench.jsonl not found in cache")
    return [json.loads(l) for l in open(path[0], encoding="utf-8") if l.strip()]


def build(out_dir: str):
    rows = load_bix()
    os.makedirs(out_dir, exist_ok=True)
    mcq_out, tf_out = [], []

    for i, r in enumerate(rows):
        q = (r.get("question") or "").strip()
        ideal = str(r.get("ideal", "")).strip()
        distractors = [str(d).strip() for d in (r.get("distractors") or [])]

        # --- MCQ form ---
        if q and ideal and distractors:
            opts = stable_shuffle([ideal] + distractors, f"{i}|{ideal}")
            letters = [chr(65 + k) for k in range(len(opts))]
            choice_block = "\n".join(f"{letters[k]}. {o}" for k, o in enumerate(opts))
            gold_letter = letters[opts.index(ideal)]
            user = (f"{q}\nThe result will be one of the following options; "
                    f"answer with the letter.\n{choice_block}")
            try:
                mcq_out.append(make_sample(
                    id=f"bixbench_mcq:{i:04d}",
                    task_family="F8_biomed_qa", task_id="bixbench_mcq",
                    modality=["text"],
                    messages=[{"role": "user", "content": user},
                              {"role": "assistant", "content": gold_letter}],
                    answer_short=gold_letter, choices=letters, metric="accuracy",
                    split="test", license="CC-BY-4.0", source="futurehouse/BixBench"))
            except Exception:
                pass

        # --- T/F form ---
        ans = str(r.get("answer", "")).strip()
        h = (r.get("hypothesis") or "").strip()
        res = (r.get("result") or "").strip()
        if ans in ("True", "False") and h and res:
            user = ("Based on the research result below, determine if the hypothesis is "
                    "True or False. The result will be one of the following: True, False.\n"
                    f"### Hypothesis:\n{h}\n\n### Research Result:\n{res[:800]}")
            try:
                tf_out.append(make_sample(
                    id=f"bixbench_tf:{i:04d}",
                    task_family="F8_biomed_qa", task_id="bixbench_tf",
                    modality=["text"],
                    messages=[{"role": "user", "content": user},
                              {"role": "assistant", "content": ans}],
                    answer_short=ans, choices=["True", "False"], metric="accuracy",
                    split="test", license="CC-BY-4.0", source="futurehouse/BixBench"))
            except Exception:
                pass

    for name, samples in [("bixbench_mcq", mcq_out), ("bixbench_tf", tf_out)]:
        path = os.path.join(out_dir, f"f8_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(s.to_json() + "\n")
        print(f"[F8 {name}] wrote {len(samples)} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    build(a.out)


if __name__ == "__main__":
    main()
