#!/usr/bin/env python
# coding: utf-8
"""
01-build_unified_jsonl.py

Merge all OmniGene-5 multi-modal datasets into a unified LLaVA-style JSONL
format. Streams large JSONs/parquets to avoid OOM.

Output: /root/autodl-tmp/dnagpt/omnigene5/data/unified/
  - train.jsonl   (one JSON object per line)
  - val.jsonl
  - meta.json     (dataset stats)

Unified record schema:
{
  "id": "<dataset>:<idx>",
  "modality": ["vision", "text"]    # or ["sequence", "text"], etc.
  "images": ["<absolute_path>"],     # may be empty list
  "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
  ],
  "task_family": "molecule" | "protein" | "microscopy" | "biomed_vqa" |
                 "chart" | "scifig" | "interaction" | "knowledge" | "structure",
  "source": "<dataset_id>",
}

Image paths are kept absolute; the trainer resolves them at __getitem__ time.
"""
import os
import json
import random
import zipfile
import tarfile
from pathlib import Path
from collections import Counter, defaultdict

import ijson
import pyarrow.parquet as pq

DATA = Path("/root/autodl-tmp/dnagpt/omnigene5/data")
OUT = DATA / "unified"
OUT.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)

# Per-source caps to balance the corpus (avoid one source dominating)
CAPS = {
    "vis_chebi20": 100000,        # 131K available, cap at 100K
    "pubmedvision_align": 50000,  # 760K available, cap
    "pubmedvision_it": 50000,
    "biomed_vis_caption": 50000,  # 500K available, cap
    "biomed_vis_synthetic": 30000,
    "chartqa": 28000,             # full
    "scicap": 0,                  # parquet/format-check first, deferred
    "hpa10m": 30000,              # 40K available
    "biomatrix_protein_1d": 50000,
    "biomatrix_protein_3d": 30000,
    "biomatrix_molecule_1d_smiles": 50000,
    "biomatrix_molecule_3d": 0,   # parquet path TBD
    "biomatrix_interaction_1d": 30000,
    "protein2text_qa": 20000,
    "wanglab_protein_catalogue": 30000,
    "im-sangwoon_uniprot": 30000,
    "wanglab_bioreason": 30000,
    "baai_opi_struc": 30000,
    "xin1222_sa_prot": 30000,
}

VAL_FRAC = 0.02

writer_train = open(OUT / "train.jsonl", "w")
writer_val = open(OUT / "val.jsonl", "w")
counts = Counter()
modality_counts = Counter()
task_counts = Counter()

written = 0


def emit(rec):
    global written
    counts[rec["source"]] += 1
    modality_counts["+".join(rec["modality"])] += 1
    task_counts[rec["task_family"]] += 1
    if random.random() < VAL_FRAC:
        json.dump(rec, writer_val); writer_val.write("\n")
    else:
        json.dump(rec, writer_train); writer_train.write("\n")
    written += 1
    if written % 10000 == 0:
        print(f"  ...{written} records written")


def std_messages(user, assistant):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ============== A. Vis-CheBI20 ==============
def load_vis_chebi20():
    src = "vis_chebi20"
    cap = CAPS[src]
    base = DATA / "B_chebi20"
    n = 0
    # Try both train and test split
    for split_name, split_file in [("train", "train.json"), ("test", "test.json")]:
        if n >= cap: break
        fpath = base / split_file
        if not fpath.exists(): continue
        try:
            with open(fpath) as f:
                items = json.load(f)
        except Exception:
            continue
        random.shuffle(items)
        for item in items:
            if n >= cap: break
            rel_imgs = item.get("images", [])
            abs_imgs = [str(base / rel) for rel in rel_imgs]
            if not all(Path(p).exists() for p in abs_imgs):
                continue  # image not extracted yet
            rec = {
                "id": f"{src}:{n}",
                "modality": ["vision", "text"],
                "images": abs_imgs,
                "messages": item["messages"],
                "task_family": "molecule",
                "source": src,
                "subtype": item.get("task_name", ""),
            }
            emit(rec)
            n += 1
    print(f"[{src}] emitted {n}")


# ============== B. PubMedVision (Alignment + IT) ==============
def load_pubmedvision_split(split_name, src_key, cap):
    fname = f"PubMedVision_{split_name}_VQA.json"
    fpath = DATA / "D_pubmedvision" / fname
    if not fpath.exists():
        print(f"[{src_key}] file missing, skip")
        return
    img_root = DATA / "D_pubmedvision"
    n = 0
    with open(fpath, "rb") as f:
        for item in ijson.items(f, "item"):
            if n >= cap: break
            rel_imgs = item.get("image", [])
            if isinstance(rel_imgs, str):
                rel_imgs = [rel_imgs]
            abs_imgs = []
            for rel in rel_imgs:
                # PubMedVision images are inside images_*.zip; we'll resolve at training time
                # For now, store as relative path
                abs_imgs.append(f"{img_root}/{rel}")
            convs = item.get("conversations", [])
            if len(convs) < 2:
                continue
            # convs is "from"/"value" format
            msgs = []
            for c in convs:
                role = "user" if c["from"] == "human" else "assistant"
                msgs.append({"role": role, "content": c["value"]})
            rec = {
                "id": f"{src_key}:{n}",
                "modality": ["vision", "text"],
                "images": abs_imgs,
                "messages": msgs,
                "task_family": "biomed_vqa",
                "source": src_key,
                "subtype": item.get("modality", "") + "/" + item.get("body_part", ""),
            }
            emit(rec)
            n += 1
    print(f"[{src_key}] emitted {n}")


# ============== C. biomed-visual-instructions ==============
def load_biomed_vis(filename, src_key, cap):
    fpath = DATA / "J_biomedvis" / filename
    if not fpath.exists(): return
    n = 0
    with open(fpath, "rb") as f:
        for item in ijson.items(f, "item"):
            if n >= cap: break
            rel_imgs = item.get("images", [])
            if isinstance(rel_imgs, str):
                rel_imgs = [rel_imgs]
            # biomed-visual stores image filenames; need to verify resolution
            abs_imgs = [f"{DATA}/J_biomedvis/{p}" for p in rel_imgs]
            msgs = item.get("messages", [])
            if not msgs:
                continue
            rec = {
                "id": f"{src_key}:{n}",
                "modality": ["vision", "text"],
                "images": abs_imgs,
                "messages": msgs,
                "task_family": "biomed_vqa",
                "source": src_key,
            }
            emit(rec)
            n += 1
    print(f"[{src_key}] emitted {n}")


# ============== D. ChartQA ==============
def load_chartqa():
    src = "chartqa"
    cap = CAPS[src]
    base = DATA / "E_chartqa" / "data"
    n = 0
    for pf in sorted(base.glob("train-*.parquet")):
        if n >= cap: break
        f = pq.ParquetFile(pf)
        for batch in f.iter_batches(batch_size=256):
            if n >= cap: break
            for row in batch.to_pylist():
                if n >= cap: break
                # 'image' is bytes; we save it on disk
                img_bytes = row.get("image", {}).get("bytes")
                if not img_bytes: continue
                img_dir = OUT / "chartqa_imgs"
                img_dir.mkdir(exist_ok=True)
                img_path = img_dir / f"{n:06d}.png"
                if not img_path.exists():
                    with open(img_path, "wb") as fp:
                        fp.write(img_bytes)
                query = row.get("query", "")
                label = row.get("label", [""])
                if isinstance(label, list): label = label[0] if label else ""
                rec = {
                    "id": f"{src}:{n}",
                    "modality": ["vision", "text"],
                    "images": [str(img_path)],
                    "messages": std_messages(f"<image>\n{query}", str(label)),
                    "task_family": "chart",
                    "source": src,
                }
                emit(rec)
                n += 1
    print(f"[{src}] emitted {n}")


# ============== E. HPA10M (webdataset format) ==============
def load_hpa10m():
    src = "hpa10m"
    cap = CAPS[src]
    base = DATA / "C_hpa_microscopy" / "hpa10m_train"
    img_dir = OUT / "hpa10m_imgs"
    img_dir.mkdir(exist_ok=True)
    n = 0
    for tar_path in sorted(base.glob("*.tar")):
        if n >= cap: break
        try:
            t = tarfile.open(tar_path, "r")
        except Exception:
            continue
        # webdataset: same key has .jpg + .json
        members = {}
        for m in t.getmembers():
            stem = m.name.rsplit(".", 1)[0]
            ext = m.name.rsplit(".", 1)[-1] if "." in m.name else ""
            members.setdefault(stem, {})[ext] = m
        for stem, parts in members.items():
            if n >= cap: break
            if "jpg" not in parts or "json" not in parts: continue
            # extract metadata
            jf = t.extractfile(parts["json"])
            if jf is None: continue
            meta = json.loads(jf.read())
            # extract image to disk (we don't keep bytes in JSONL)
            img_path = img_dir / f"{stem.replace('/', '_')}.jpg"
            if not img_path.exists():
                jpg_f = t.extractfile(parts["jpg"])
                if jpg_f is None: continue
                with open(img_path, "wb") as fp:
                    fp.write(jpg_f.read())
            caption = meta.get("caption_1") or meta.get("generic_caption") or ""
            gene = meta.get("gene_name", "")
            cell_type = meta.get("cell_type", "")
            user = (
                "<image>\nDescribe this Human Protein Atlas image. "
                f"What protein and cell type does it show?"
            )
            assistant = (
                f"This image shows the protein {gene} in {cell_type} cells. "
                f"{caption}"
            )
            rec = {
                "id": f"{src}:{n}",
                "modality": ["vision", "text"],
                "images": [str(img_path)],
                "messages": std_messages(user, assistant),
                "task_family": "microscopy",
                "source": src,
                "subtype": meta.get("category", ""),
            }
            emit(rec)
            n += 1
        t.close()
    print(f"[{src}] emitted {n}")


# ============== F. BioMatrix-SFT (Alpaca format, parquet) ==============
def load_biomatrix(subdir, src_key, task_family, cap):
    base = DATA / "H_biomatrix" / subdir
    if not base.exists(): return
    n = 0
    for pf in sorted(base.glob("train-*.parquet")):
        if n >= cap: break
        f = pq.ParquetFile(pf)
        for batch in f.iter_batches(batch_size=256):
            if n >= cap: break
            for row in batch.to_pylist():
                if n >= cap: break
                instr = row.get("instruction", "")
                inp = row.get("input", "")
                out = row.get("output", "")
                if not instr or not out: continue
                user = f"{instr}\n\n{inp}".strip() if inp else instr
                rec = {
                    "id": f"{src_key}:{n}",
                    "modality": ["text"],
                    "images": [],
                    "messages": std_messages(user, out),
                    "task_family": task_family,
                    "source": src_key,
                }
                emit(rec)
                n += 1
    print(f"[{src_key}] emitted {n}")


# ============== G. Protein QA & annotation datasets ==============
def load_protein2text_qa():
    src = "protein2text_qa"
    cap = CAPS[src]
    base = DATA / "L_tumorailab_Protein2Text-QA"
    n = 0
    for fname in [
        "protein2text_QA_intro_set_long_format.json",
        "protein2text_QA_discussion_set_long_format.json",
        "protein2text_QA_test_split_long_format.json",
    ]:
        fpath = base / fname
        if not fpath.exists(): continue
        if n >= cap: break
        with open(fpath, "rb") as f:
            for item in ijson.items(f, "item"):
                if n >= cap: break
                seq = item.get("amino_seq", "")
                convs = item.get("conversations", [])
                if len(convs) < 2: continue
                # convs use {from: human/gpt, value: text} format
                msgs = []
                for c in convs:
                    role = "user" if c.get("from") == "human" else "assistant"
                    val = c.get("value", "")
                    # First user turn has <protein_sequence> placeholder; expand it
                    if role == "user" and "<protein_sequence>" in val:
                        val = val.replace("<protein_sequence>", f"Protein sequence:\n{seq}\n")
                    msgs.append({"role": role, "content": val})
                rec = {
                    "id": f"{src}:{n}",
                    "modality": ["sequence", "text"],
                    "images": [],
                    "messages": msgs,
                    "task_family": "knowledge",
                    "source": src,
                }
                emit(rec)
                n += 1
    print(f"[{src}] emitted {n}")


# ============== Run all ==============
print("\n=== Vis-CheBI20 ===")
load_vis_chebi20()

print("\n=== PubMedVision Alignment ===")
load_pubmedvision_split("Alignment", "pubmedvision_align", CAPS["pubmedvision_align"])

print("\n=== PubMedVision IT ===")
load_pubmedvision_split("InstructionTuning", "pubmedvision_it", CAPS["pubmedvision_it"])

print("\n=== biomed-visual-instructions caption ===")
load_biomed_vis("image_caption_pairs.json", "biomed_vis_caption", CAPS["biomed_vis_caption"])

print("\n=== biomed-visual-instructions synthetic ===")
load_biomed_vis("synthetic_visual_tasks.json", "biomed_vis_synthetic", CAPS["biomed_vis_synthetic"])

print("\n=== ChartQA ===")
load_chartqa()

print("\n=== HPA10M ===")
load_hpa10m()

print("\n=== BioMatrix protein_1d ===")
load_biomatrix("protein_1d", "biomatrix_protein_1d", "protein", CAPS["biomatrix_protein_1d"])

print("\n=== BioMatrix protein_3d ===")
load_biomatrix("protein_3d", "biomatrix_protein_3d", "structure", CAPS["biomatrix_protein_3d"])

print("\n=== BioMatrix molecule_1d_smiles ===")
load_biomatrix("molecule_1d_smiles", "biomatrix_molecule_1d_smiles", "molecule", CAPS["biomatrix_molecule_1d_smiles"])

print("\n=== BioMatrix interaction_1d ===")
load_biomatrix("interaction_1d", "biomatrix_interaction_1d", "interaction", CAPS["biomatrix_interaction_1d"])

print("\n=== Protein2Text-QA ===")
load_protein2text_qa()


# ============== Finalize ==============
writer_train.close()
writer_val.close()

meta = {
    "total_records": written,
    "by_source": dict(counts),
    "by_modality": dict(modality_counts),
    "by_task_family": dict(task_counts),
    "val_fraction": VAL_FRAC,
    "seed": SEED,
}
with open(OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n=== DONE ===")
print(f"Total records: {written}")
print(f"By source: {dict(counts)}")
print(f"By modality: {dict(modality_counts)}")
print(f"By task family: {dict(task_counts)}")
print(f"Output: {OUT}")
