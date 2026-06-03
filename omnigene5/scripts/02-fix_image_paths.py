#!/usr/bin/env python
# coding: utf-8
"""
02-fix_image_paths.py

After 01-build_unified_jsonl.py, fix image paths so that:
- PubMedVision images point to images_extracted/images/<basename>
- biomed_vis_* (which references PubMedVision images) also points there
- HPA10M paths are kept (already extracted)
- Vis-CheBI20 paths are kept (test/* already extracted)

Filter out records whose images do not exist on disk.

Output: replaces train.jsonl / val.jsonl in-place.
"""
import json
import os
import shutil
from pathlib import Path

UNIFIED = Path("/root/autodl-tmp/dnagpt/omnigene5/data/unified")
PMC_REAL = Path("/root/autodl-tmp/dnagpt/omnigene5/data/D_pubmedvision/images_extracted/images")
BIOMED_DEFAULT = Path("/root/autodl-tmp/dnagpt/omnigene5/data/J_biomedvis")

# Build a set of existing PMC images
print("Indexing real images...", flush=True)
pmc_existing = set(os.listdir(PMC_REAL)) if PMC_REAL.exists() else set()
print(f"  PubMedVision extracted: {len(pmc_existing)}")


def fix_path(p):
    """Resolve image path. If it's a PubMedVision-style pmc_X.jpg, redirect to extracted dir."""
    base = os.path.basename(p)
    if base.startswith("pmc_") and base.endswith(".jpg"):
        candidate = PMC_REAL / base
        if base in pmc_existing:
            return str(candidate)
    # Otherwise check if path exists as-is
    if os.path.exists(p):
        return p
    return None  # broken


for split in ("train.jsonl", "val.jsonl"):
    src_path = UNIFIED / split
    bak_path = UNIFIED / f"{split}.bak"
    if not bak_path.exists():
        shutil.copy(src_path, bak_path)
    print(f"\n=== {split} ===")
    n_in = 0
    n_out = 0
    n_dropped = 0
    drop_reasons = {}
    with open(bak_path) as f, open(src_path, "w") as out:
        for line in f:
            n_in += 1
            r = json.loads(line)
            if r["images"]:
                fixed_imgs = []
                for p in r["images"]:
                    fp = fix_path(p)
                    if fp is None:
                        break
                    fixed_imgs.append(fp)
                if len(fixed_imgs) != len(r["images"]):
                    n_dropped += 1
                    drop_reasons[r["source"]] = drop_reasons.get(r["source"], 0) + 1
                    continue
                r["images"] = fixed_imgs
            out.write(json.dumps(r) + "\n")
            n_out += 1
    print(f"  in: {n_in:,}, out: {n_out:,}, dropped: {n_dropped:,}")
    if drop_reasons:
        print(f"  drop by source:")
        for k, v in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")

# Update meta
print("\nUpdating meta...")
new_train = sum(1 for _ in open(UNIFIED / "train.jsonl"))
new_val = sum(1 for _ in open(UNIFIED / "val.jsonl"))
print(f"  train: {new_train:,}, val: {new_val:,}, total: {new_train+new_val:,}")
print("Done.")
