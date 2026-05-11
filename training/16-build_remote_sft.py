#!/usr/bin/env python
# coding: utf-8
"""
16-build_remote_sft.py
构造 remote-homology 强化 SFT 增量数据。

策略:
- 从 dnagpt/biopaws::protein_pair_remote 采样 10k pos + 10k neg
- 排除 evaluate (seed=42) 用过的 2000 条
- 输出与已有 SFT 格式一致: instruction/input/output (Homologous / Non-Homologous)
- 追加到 omnigene_sft_v1_train.jsonl, 生成新的 *_with_remote.jsonl
"""
import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
import json
import random
from pathlib import Path
from datasets import load_dataset

SEED_EVAL = 42       # 评测使用的 seed, 要排除这批
SEED_SFT = 2027      # 新 SFT 采样 seed
N_PER_CLASS = 10000

TRAIN_DIR = Path("/root/autodl-fs/omnigene_v2/sft_data/train")
OUT_PATH = TRAIN_DIR / "omnigene_sft_v1_train_with_remote.jsonl"
BASE_TRAIN = TRAIN_DIR / "omnigene_sft_v1_train.jsonl"

# 1. 拉远缘同源
print("Loading protein_pair_remote...", flush=True)
ds = load_dataset('dnagpt/biopaws', 'protein_pair_remote')['train']
print(f"  rows: {len(ds)}")

records = list(ds)
label0 = [r for r in records if int(r['label']) == 0]
label1 = [r for r in records if int(r['label']) == 1]
print(f"  label0: {len(label0)}, label1: {len(label1)}")

# 2. 重放 evaluate 采样，拿到已用样本的指纹 -> 后续排除
random.seed(SEED_EVAL)
r0 = random.sample(label0, min(1000, len(label0)))
r1 = random.sample(label1, min(1000, len(label1)))
used = {(x['sentence1'], x['sentence2']) for x in r0 + r1}
print(f"  excluded (evaluate set): {len(used)}")

# 3. 从剩余池子采样
remain0 = [r for r in label0 if (r['sentence1'], r['sentence2']) not in used]
remain1 = [r for r in label1 if (r['sentence1'], r['sentence2']) not in used]
print(f"  remaining label0/1: {len(remain0)}/{len(remain1)}")

random.seed(SEED_SFT)
pick0 = random.sample(remain0, N_PER_CLASS)
pick1 = random.sample(remain1, N_PER_CLASS)
picked = pick0 + pick1
random.shuffle(picked)
print(f"  picked for SFT: {len(picked)}")

# 4. 构造 instruction/input/output
INSTR_POOL = [
    "Determine if the two sequences below are structurally related (like paraphrases).",
    "Decide whether these two protein sequences are homologous.",
    "Judge if the following two protein sequences share an evolutionary origin.",
    "Are the two sequences below homologous? Answer with 'Homologous' or 'Non-Homologous'.",
    "Given two protein sequences, classify them as Homologous or Non-Homologous.",
]

def make_record(r, rng):
    instr = rng.choice(INSTR_POOL)
    inp = f"### Sequence 1:\n{r['sentence1']}\n\n### Sequence 2:\n{r['sentence2']}"
    out = "Homologous" if int(r['label']) == 1 else "Non-Homologous"
    return {"instruction": instr, "input": inp, "output": out}

rng = random.Random(SEED_SFT + 1)
new_records = [make_record(r, rng) for r in picked]

# 5. 追加到已有 train
print(f"\nReading base train: {BASE_TRAIN}")
with BASE_TRAIN.open() as f:
    base_lines = f.readlines()
print(f"  base rows: {len(base_lines)}")

with OUT_PATH.open("w") as f:
    f.writelines(base_lines)
    for r in new_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nWrote {OUT_PATH}")
print(f"  total rows: {len(base_lines) + len(new_records)} "
      f"(base {len(base_lines)} + remote {len(new_records)})")
print(f"  size: {OUT_PATH.stat().st_size / 1024**2:.1f} MB")
