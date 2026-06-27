"""F9 converter: reuse the OmniGene-4-MM unified corpus (Vis-CheBI20 etc.) -> BioPAWS-2 QA.

The MM corpus (omnigene5/data/unified/{train,val}.jsonl) is ALREADY in messages format.
This converter selects the molecule-image records (vis_chebi20), normalizes them to the
BioPAWS-2 schema (adds task_family/answer_short/metric/license/source), and re-splits.

For struct_recog (which functional group is highlighted) the answer is a class -> accuracy.
For struct_cap / trans_iupac (free-form description) -> rouge_l / bleu.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from schema import make_sample  # noqa: E402

UNIFIED = "/root/autodl-tmp/dnagpt/omnigene5/data/unified"

# subtype -> (task_id, metric, answer-extraction)
SUBTYPE_MAP = {
    "struct_recog": ("mol_recog", "accuracy"),
    "struct_cap": ("mol_caption", "rouge_l"),
    "trans_iupac": ("mol_iupac", "bleu"),
}


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def _answer_short(subtype: str, assistant: str) -> str:
    """Pull a canonical short answer for scoring."""
    if subtype == "struct_recog":
        # "The functional group benzene is highlighted." -> benzene
        toks = assistant.replace(".", "").split()
        if "group" in toks:
            i = toks.index("group")
            if i + 1 < len(toks):
                return toks[i + 1]
        return assistant.strip()
    # free-form: use full text as gold reference for rouge/bleu
    return assistant.strip()


def build(out_dir: str, max_n: int | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "f9_vis_chebi20.jsonl")
    n = 0
    seen_subtypes: dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as out:
        for fname in ("train.jsonl", "val.jsonl"):
            path = os.path.join(UNIFIED, fname)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("source") != "vis_chebi20":
                        continue
                    subtype = rec.get("subtype", "")
                    if subtype not in SUBTYPE_MAP:
                        continue
                    if max_n and n >= max_n:
                        break
                    task_id, metric = SUBTYPE_MAP[subtype]
                    msgs = rec["messages"]
                    assistant = msgs[-1]["content"]
                    ans = _answer_short(subtype, assistant)
                    imgs = rec.get("images", [])
                    # prepend <image> placeholder to the user turn if missing
                    user_msg = dict(msgs[0])
                    if imgs and "<image>" not in user_msg["content"]:
                        user_msg["content"] = "<image>\n" + user_msg["content"]
                    uid = f"f9_{task_id}:{n:06d}"
                    s = make_sample(
                        id=uid,
                        task_family="F9_multimodal",
                        task_id=task_id,
                        modality=["vision", "text"],
                        images=list(imgs),
                        messages=[user_msg, {"role": "assistant", "content": assistant}],
                        answer_short=ans,
                        metric=metric,
                        split=split_of(uid),
                        license="CC-BY-4.0",
                        source="vis_chebi20",
                    )
                    out.write(s.to_json() + "\n")
                    n += 1
                    seen_subtypes[subtype] = seen_subtypes.get(subtype, 0) + 1

    print(f"[F9 vis_chebi20] wrote {n} samples -> {out_path}")
    print(f"               subtypes: {seen_subtypes}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    build(a.out, a.max)


if __name__ == "__main__":
    main()
