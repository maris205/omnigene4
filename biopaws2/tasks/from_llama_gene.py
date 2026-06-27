"""Converter: llama-gene SFT data (dnagpt/llama-gene-train-data) -> BioPAWS-2 QA jsonl.

The llama-gene corpus already uses exactly the format the user asked for: the instruction
lists the full candidate-label set ("The result will be one of the following: A, B, C.")
and the output is one label. This is the general multi-class QA form (superior to Yes/No):
it works for 2-class AND k-class tasks, and forces label-grounded answering.

This converter:
  - parses {instruction, input, output} JSONL,
  - extracts the candidate label list from the instruction tail,
  - groups by task (Fold classes, Subcellular Location, promoter, splice site, Central
    Dogma, signal peptide, TF, ...),
  - emits schema-valid `messages` records with `choices` = the candidate labels and
    `answer_short` = the gold label,
  - assigns family (F2/F3/F5/F6) by task, metric=accuracy (f1 for multi-class),
  - deterministic 80/10/10 split (llama-gene ships train + eva; we keep its eva as test
    and re-split train into train/val).

Usage:
  python -m tasks.from_llama_gene --out data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import make_sample  # noqa: E402

LG_DIR = "/root/autodl-tmp/dnagpt/biopaws2/data/llama_gene"

# map a normalized task key -> (task_id, family, modality, metric)
TASK_MAP = {
    "promoter detection": ("promoter_detection", "F3_dna", ["dna", "text"], "accuracy"),
    "core promoter detection": ("core_promoter_detection", "F3_dna", ["dna", "text"], "accuracy"),
    "splice site prediction": ("splice_site", "F3_dna", ["dna", "text"], "accuracy"),
    "transcription factor prediction": ("tf_prediction", "F3_dna", ["dna", "text"], "accuracy"),
    "fold classes": ("fold_class", "F5_structure", ["protein", "text"], "f1"),
    "prokaryotic protein subcellular location": ("subcellular_loc", "F2_functional", ["protein", "text"], "f1"),
    "signal peptides": ("signal_peptide", "F2_functional", ["protein", "text"], "accuracy"),
    "immensely diverse nps": ("npp", "F2_functional", ["protein", "text"], "accuracy"),
    "central dogma": ("central_dogma", "F6_crossmodal", ["dna", "protein", "text"], "accuracy"),
}


def task_key(instruction: str) -> str:
    instr = instruction.lower()
    if "central dogma" in instr:
        return "central dogma"
    m = re.search(r"determine (?:the )?(.+?) of following", instr)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())
    return ""


def parse_choices(instruction: str) -> list[str]:
    """Extract candidate labels from 'one of the following: A,B,C.'"""
    m = re.search(r"one of the following:\s*(.+?)\.?\s*$", instruction.strip())
    if not m:
        return []
    raw = m.group(1)
    parts = [p.strip(" .") for p in re.split(r"[,，]", raw) if p.strip(" .")]
    return parts


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 88 else "val"  # eva files become test; train→train/val 88/12


def load_jsonl(path):
    rows = []
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  (skipped {bad} malformed lines in {os.path.basename(path)})")
    return rows


def convert(rows, force_split=None, src=""):
    out = []
    skipped = 0
    for i, r in enumerate(rows):
        instr = r.get("instruction", "")
        inp = r.get("input", "")
        gold = r.get("output", "").strip()
        key = task_key(instr)
        if key not in TASK_MAP:
            skipped += 1
            continue
        task_id, family, modality, metric = TASK_MAP[key]
        choices = parse_choices(instr)
        if not choices or gold not in choices:
            # tolerate case/space mismatch
            norm = {c.lower().replace(" ", ""): c for c in choices}
            g2 = norm.get(gold.lower().replace(" ", ""))
            if g2:
                gold = g2
            else:
                skipped += 1
                continue
        # build user turn: instruction + sequence input (llama-gene style)
        user = instr
        if inp.strip():
            user = f"{instr}\n{inp.strip()}"
        split = force_split or split_of(f"{task_id}|{inp[:50]}|{gold}|{i}")
        uid = f"{task_id}:{src}:{i:06d}"
        try:
            s = make_sample(
                id=uid, task_family=family, task_id=task_id, modality=modality,
                messages=[{"role": "user", "content": user},
                          {"role": "assistant", "content": gold}],
                answer_short=gold, choices=choices, metric=metric, split=split,
                license="Apache-2.0", source=f"llama-gene:{src}")
            out.append(s)
        except Exception:
            skipped += 1
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    files = {
        "dna_train": ("dna_sft_dna_train.json", None),
        "dna_eva": ("dna_sft_dna_eva.json", "test"),
        "prot_train": ("protein_protein_sft_train.json", None),
        "prot_eva": ("protein_sft_protein_eva.json", "test"),
    }
    # collect by task_id, then write one jsonl per task
    by_task: dict[str, list] = {}
    for src, (fname, force) in files.items():
        path = os.path.join(LG_DIR, fname)
        if not os.path.exists(path):
            print(f"  missing {path}, skip")
            continue
        rows = load_jsonl(path)
        samples, skipped = convert(rows, force_split=force, src=src)
        print(f"[{src}] {len(rows)} rows -> {len(samples)} samples ({skipped} skipped)")
        for s in samples:
            by_task.setdefault(s.task_id, []).append(s)

    os.makedirs(a.out, exist_ok=True)
    print("\n=== per-task output ===")
    for task_id, samples in sorted(by_task.items()):
        path = os.path.join(a.out, f"lg_{task_id}.jsonl")
        from collections import Counter
        sp = Counter(s.split for s in samples)
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(s.to_json() + "\n")
        print(f"  {task_id}: {len(samples)} -> {path}  splits={dict(sp)}")


if __name__ == "__main__":
    main()
