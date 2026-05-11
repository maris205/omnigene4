#!/usr/bin/env python
# coding: utf-8
"""
12-merge_sft_v2.py

按 cell-mul.md 3.3 节的比例合成 OmniGene-SFT-v1:
- master: 全量 + metadata
- train:  只留 instruction/input/output
- eval:   结构化保留 label + metadata (从 master 里 shuffle 抽 2k)

来源:
- 已有 (messages 格式): homology / structure / UniProtQA / MutaDescribe
- 新增 (instruction 格式): cell / mol
"""
import os
import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

SFT_DIR = Path("/root/autodl-fs/omnigene_v2/sft_data")
OLD_DIR = Path("/root/autodl-tmp/dnagpt/biopaws/sft_data")
MASTER_DIR = SFT_DIR / "master"
TRAIN_DIR = SFT_DIR / "train"
EVAL_DIR = SFT_DIR / "eval"
STATS_DIR = SFT_DIR / "stats"
for d in [MASTER_DIR, TRAIN_DIR, EVAL_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_N = 200000
MIX = {
    "DNA": 0.02,       # 池子太小，缩到 ~4k
    "Protein": 0.25,
    "Structure": 0.10,
    "Literature": 0.20,
    "Mutation": 0.15,
    "Cell": 0.15,
    "Mol": 0.13,
    "General": 0.00,   # 只有 4 条，干脆丢进 Literature
}

# 映射: 已有 jsonl -> (类别, instruction 转换方式)
SOURCES = {
    # 格式: (path, category, source_format)
    OLD_DIR / "homology_task_sft.jsonl": ("Protein", "messages"),
    OLD_DIR / "structure_task_sft.jsonl": ("Structure", "messages"),
    OLD_DIR / "uniprotqa_sft.jsonl": ("Literature", "messages"),
    OLD_DIR / "mutadescribe_sft.jsonl": ("Mutation", "messages"),
    OLD_DIR / "nl_bio_sft.jsonl": ("General", "messages"),
    MASTER_DIR / "cell_sft_master.jsonl": ("Cell", "instruction"),
    MASTER_DIR / "mol_sft_master.jsonl": ("Mol", "instruction"),
}


def convert_messages_to_instruction(rec, category):
    """把 {messages: [{user}, {assistant}]} 转成 {instruction, input, output}."""
    msgs = rec.get("messages", [])
    user_content = ""
    assistant_content = ""
    for m in msgs:
        if m["role"] == "user":
            user_content = m["content"]
        elif m["role"] == "assistant":
            assistant_content = m["content"]
    # 尝试从 user 里切出 instruction + input
    instr, inp = user_content, ""
    # 常见模板: "### Instruction:\n...\n### Sentence 1:\n..."
    markers = ["### Sequence 1:", "### Sentence 1:", "### Protein Sequence:",
               "### Wild-type:", "### Sequence:", "### Sequence 1D:"]
    for mk in markers:
        if mk in user_content:
            parts = user_content.split(mk, 1)
            instr = parts[0].strip().replace("### Instruction:", "").strip()
            inp = (mk + parts[1]).strip()
            break
    # 清理 assistant: 去掉 THOUGHT 块
    import re
    out = re.sub(r"<THOUGHT>.*?</THOUGHT>\s*", "", assistant_content,
                 flags=re.DOTALL).strip()
    if not out:
        out = assistant_content.strip()
    # 若 instr 还带 "### Answer:" 末尾, 去掉
    instr = instr.replace("### Answer:", "").strip()
    if not instr:
        instr = "Answer the question based on the input."
    return {
        "instruction": instr,
        "input": inp,
        "output": out,
    }


# 1. 按源读入 + 归类
buckets = {cat: [] for cat in MIX}
for fp, (category, fmt) in SOURCES.items():
    if not fp.exists():
        print(f"  skip missing {fp}")
        continue
    n = 0
    with fp.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if fmt == "messages":
                trio = convert_messages_to_instruction(rec, category)
                md = {"source_file": fp.name, "type": rec.get("type"),
                      "source": rec.get("source")}
            else:  # instruction already
                trio = {k: rec[k] for k in ("instruction", "input", "output")}
                md = {"source_file": fp.name, "task": rec.get("task"),
                      **(rec.get("metadata") or {})}
            buckets[category].append({"trio": trio, "metadata": md,
                                       "category": category})
            n += 1
    print(f"  loaded {n} from {fp.name} -> {category} (now {len(buckets[category])})")

# 2. 特殊处理: DNA 目前没有独立文件, 用 homology + structure 里的 DNA 任务分一部分
# 这里暂时先不硬凑, 直接看每个 bucket 量
print("\nBucket sizes before sampling:")
for k, v in buckets.items():
    print(f"  {k:<12s}: {len(v)}")

# 由于 Protein 量很大, 其他不够, 按比例需要从 Protein 借 DNA
# 这里简单处理: 如果某 bucket 为空且在 MIX 里, 按文本匹配从其他 bucket 借
if not buckets["DNA"]:
    # 从 structure/protein 里找 DNA 相关的样本
    moved = []
    for src in ["Structure", "Protein"]:
        keep = []
        for it in buckets[src]:
            text = (it["trio"]["instruction"] + " " + it["trio"]["input"]).lower()
            if ("dna" in text or "genome" in text or "nucleotid" in text) and len(moved) < 5000:
                moved.append(it)
            else:
                keep.append(it)
        buckets[src] = keep
    buckets["DNA"] = moved
    print(f"  DNA bucket filled from moves: {len(moved)}")

print("\nBucket sizes after rebalance:")
for k, v in buckets.items():
    print(f"  {k:<12s}: {len(v)}")

# 3. 按比例采样
quotas = {k: int(TARGET_N * v) for k, v in MIX.items()}
diff = TARGET_N - sum(quotas.values())
if diff:
    quotas["Protein"] += diff

final = []
actual_counts = {}
for cat, q in quotas.items():
    pool = buckets.get(cat, [])
    if not pool:
        print(f"  WARN: category {cat} is empty, skip")
        actual_counts[cat] = 0
        continue
    if len(pool) >= q:
        picked = random.sample(pool, q)
    else:
        # 池子不够, 放回抽样凑
        picked = pool[:]
        picked.extend(random.choices(pool, k=q - len(pool)))
    actual_counts[cat] = len(picked)
    for item in picked:
        final.append(item)

random.shuffle(final)
print(f"\nFinal total: {len(final)}")
for k in MIX:
    print(f"  {k:<12s}: {actual_counts.get(k, 0)}")

# 4. 去重 (按 instruction + input hash)
import hashlib
seen_hash = set()
dedup = []
for it in final:
    key = hashlib.md5(
        (it["trio"]["instruction"] + "||" + it["trio"]["input"] + "||" + it["trio"]["output"]).encode()
    ).hexdigest()
    if key in seen_hash:
        continue
    seen_hash.add(key)
    dedup.append(it)
print(f"After dedup: {len(dedup)} (removed {len(final) - len(dedup)})")

# 5. 写 master
master_path = MASTER_DIR / "omnigene_sft_v1_master.jsonl"
with master_path.open("w") as f:
    for i, it in enumerate(dedup):
        rec = {
            "id": f"omnigene_{i:06d}",
            "category": it["category"],
            **it["trio"],
            "metadata": it["metadata"],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"  master -> {master_path} ({master_path.stat().st_size / 1024**2:.1f} MB)")

# 6. 写 train (只保留 instruction/input/output)
# 抽 2k 做 eval, 其余全部做 train
random.shuffle(dedup)
eval_n = 2000
eval_set = dedup[:eval_n]
train_set = dedup[eval_n:]

train_path = TRAIN_DIR / "omnigene_sft_v1_train.jsonl"
with train_path.open("w") as f:
    for it in train_set:
        f.write(json.dumps(it["trio"], ensure_ascii=False) + "\n")
print(f"  train  -> {train_path} ({len(train_set)} rows, "
      f"{train_path.stat().st_size / 1024**2:.1f} MB)")

# 7. 写 eval (带 metadata + 分类)
eval_path = EVAL_DIR / "omnigene_sft_v1_eval.jsonl"
with eval_path.open("w") as f:
    for i, it in enumerate(eval_set):
        rec = {
            "id": f"eval_{i:05d}",
            "category": it["category"],
            **it["trio"],
            "metadata": it["metadata"],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"  eval   -> {eval_path} ({len(eval_set)} rows, "
      f"{eval_path.stat().st_size / 1024**2:.1f} MB)")

# 8. 统计报告
report = {
    "total_rows": len(dedup),
    "train_rows": len(train_set),
    "eval_rows": len(eval_set),
    "category_counts_after_dedup": dict(Counter(it["category"] for it in dedup)),
    "target_mix": MIX,
    "target_quota": quotas,
    "actual_before_dedup": actual_counts,
}
with (STATS_DIR / "data_mix_report.json").open("w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport -> {STATS_DIR / 'data_mix_report.json'}")
print(json.dumps(report["category_counts_after_dedup"], indent=2))
