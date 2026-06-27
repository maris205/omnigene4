"""Integrate locally-cached well-known protein datasets into BioPAWS-2 chat QA.

Sources (all already in omnigene5/data/, no download):
  - im-sangwoon/protein-sft-uniprot : UniProt knowledge QA (name/family/localization/function)
      -> F2_functional / task_id=uniprot_qa  (protein-knowledge QA; name-grounded)
  - tumorailab/Protein2Text-QA      : sequence-grounded literature QA (amino_seq + question)
      -> F2_functional / task_id=protein2text_qa  (sequence-grounded, free-form)
  - BAAI/OPI-Struc Function         : sequence -> UniProt function description
      -> F2_functional / task_id=opi_function  (sequence-grounded generation)

All emit schema-valid `messages`. Free-form answers use rouge_l; knowledge QA uses
exact-match-ish accuracy on short answers. Deterministic 80/10/10 split.
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

DATA = "/root/autodl-tmp/dnagpt/omnigene5/data"


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def conv_uniprot_qa(out_dir, max_n):
    f = f"{DATA}/L_im-sangwoon_protein-sft-uniprot/protein_sft.jsonl"
    n = 0
    from collections import Counter
    sp = Counter()
    out = os.path.join(out_dir, "int_uniprot_qa.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for i, line in enumerate(open(f, encoding="utf-8")):
            if max_n and n >= max_n:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r.get("messages")
            if not msgs or len(msgs) < 2:
                continue
            ans = msgs[-1]["content"].strip()
            split = split_of(f"uniprot|{r.get('protein_id','')}|{i}")
            try:
                s = make_sample(
                    id=f"uniprot_qa:{i:07d}",
                    task_family="F2_functional", task_id="uniprot_qa",
                    modality=["protein", "text"],
                    messages=[{"role": "user", "content": msgs[0]["content"]},
                              {"role": "assistant", "content": ans}],
                    answer_short=ans[:200], metric="rouge_l", split=split,
                    license="CC-BY-4.0", source=f"UniProtQA:{r.get('protein_id','')}")
                fh.write(s.to_json() + "\n"); n += 1; sp[split] += 1
            except Exception:
                pass
    print(f"[int uniprot_qa] {n} -> {out} {dict(sp)}")


def conv_protein2text(out_dir, max_n):
    f = f"{DATA}/L_tumorailab_Protein2Text-QA/protein2text_QA_discussion_set_long_format.json"
    if not os.path.exists(f):
        print("  protein2text missing, skip"); return
    d = json.load(open(f, encoding="utf-8"))
    n = 0
    from collections import Counter
    sp = Counter()
    out = os.path.join(out_dir, "int_protein2text_qa.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(d):
            if max_n and n >= max_n:
                break
            conv = r.get("conversations", [])
            seq = r.get("amino_seq", "")
            if len(conv) < 2 or not seq:
                continue
            q = conv[0]["value"].replace("<protein_sequence>", "").strip()
            a = conv[1]["value"].strip()
            user = f"Protein sequence: {seq}\n{q}"
            split = split_of(f"p2t|{r.get('id', i)}")
            try:
                s = make_sample(
                    id=f"protein2text_qa:{i:05d}",
                    task_family="F2_functional", task_id="protein2text_qa",
                    modality=["protein", "text"],
                    messages=[{"role": "user", "content": user},
                              {"role": "assistant", "content": a}],
                    answer_short=a[:200], metric="rouge_l", split=split,
                    license="CC-BY-4.0", source=f"Protein2Text-QA:{r.get('id', i)}")
                fh.write(s.to_json() + "\n"); n += 1; sp[split] += 1
            except Exception:
                pass
    print(f"[int protein2text_qa] {n} -> {out} {dict(sp)}")


def conv_opi_function(out_dir, max_n):
    files = glob.glob(f"{DATA}/L_BAAI_OPI-Struc/Function/**/*.json", recursive=True)
    seen = set()
    n = 0
    from collections import Counter
    sp = Counter()
    out = os.path.join(out_dir, "int_opi_function.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for path in files:
            try:
                d = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            for r in (d if isinstance(d, list) else [d]):
                if max_n and n >= max_n:
                    break
                seq = r.get("sequence", "")
                func = r.get("function", "")
                sid = r.get("swissprot_id", "")
                if not seq or not func or sid in seen:
                    continue
                seen.add(sid)
                user = (f"Describe the biological function of the following protein "
                        f"based on its sequence.\nSequence: {seq}")
                split = split_of(f"opi|{sid}")
                try:
                    s = make_sample(
                        id=f"opi_function:{n:05d}",
                        task_family="F2_functional", task_id="opi_function",
                        modality=["protein", "text"],
                        messages=[{"role": "user", "content": user},
                                  {"role": "assistant", "content": func}],
                        answer_short=func[:200], metric="rouge_l", split=split,
                        license="CC-BY-4.0", source=f"OPI-Struc:{sid}")
                    fh.write(s.to_json() + "\n"); n += 1; sp[split] += 1
                except Exception:
                    pass
    print(f"[int opi_function] {n} -> {out} {dict(sp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--uniprot-max", type=int, default=60000,
                    help="cap UniProt QA (1.55M total; cap for balance)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conv_uniprot_qa(a.out, a.uniprot_max)
    conv_protein2text(a.out, None)
    conv_opi_function(a.out, None)


if __name__ == "__main__":
    main()
