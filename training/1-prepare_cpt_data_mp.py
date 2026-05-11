#!/usr/bin/env python
# coding: utf-8
"""
CPT v2 数据准备 — 多进程版
每个数据源独立进程 tokenize，最后合并
"""

import os
import random
import json
import struct
import numpy as np
from multiprocessing import Process, Queue
from transformers import AutoTokenizer

DATA_DIR = os.getenv("DATA_DIR", "/root/autodl-fs/omnigene_v2/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/root/autodl-fs/omnigene_v2/models/gemma-4-26B-A4B-it-bio")
OUT_DIR = os.getenv("OUT_DIR", "/root/autodl-fs/omnigene_v2/cpt_data/data_v2")
LIT_FILE = os.getenv("LIT_FILE", "/root/autodl-fs/omnigene_v2/literature/s2orc_biology_text.txt")
REPLAY_FILE = os.getenv("REPLAY_FILE", "/root/autodl-fs/omnigene_v2/sft_data/bio_sft_v2_train.jsonl")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
MAX_LENGTH = 1024
MIN_LENGTH = 64

def stream_tokenize_file(filepath, target_bytes, sample_rate, label, out_file, seed=42):
    """独立进程：流式读取 + tokenize + 写入二进制"""
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    written_bytes = 0
    total_chunks = 0
    total_tokens = 0
    buf = []
    buf_len = 0

    with open(filepath, "r", errors="ignore") as fin, open(out_file, "wb") as fout:
        for line in fin:
            if random.random() > sample_rate:
                continue
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            buf_len += len(line)

            if buf_len >= MAX_LENGTH * 4:
                text = " ".join(buf)
                ids = tokenizer.encode(text, add_special_tokens=False)
                for j in range(0, len(ids), MAX_LENGTH):
                    chunk = ids[j:j + MAX_LENGTH]
                    if len(chunk) >= MIN_LENGTH:
                        arr = np.array(chunk, dtype=np.uint32)
                        fout.write(struct.pack("I", len(chunk)))
                        fout.write(arr.tobytes())
                        total_chunks += 1
                        total_tokens += len(chunk)
                written_bytes += buf_len
                buf = []
                buf_len = 0
                if total_chunks % 50000 == 0 and total_chunks > 0:
                    print(f"  [{label}] {total_chunks} chunks, {written_bytes/1024**3:.2f}GB", flush=True)
            if written_bytes >= target_bytes:
                break

        if buf:
            text = " ".join(buf)
            ids = tokenizer.encode(text, add_special_tokens=False)
            for j in range(0, len(ids), MAX_LENGTH):
                chunk = ids[j:j + MAX_LENGTH]
                if len(chunk) >= MIN_LENGTH:
                    arr = np.array(chunk, dtype=np.uint32)
                    fout.write(struct.pack("I", len(chunk)))
                    fout.write(arr.tobytes())
                    total_chunks += 1
                    total_tokens += len(chunk)

    print(f"  [{label}] DONE: {total_chunks} chunks, {total_tokens:,} tokens, {written_bytes/1024**3:.2f}GB", flush=True)
    # 写统计
    with open(out_file + ".stats", "w") as f:
        json.dump({"label": label, "chunks": total_chunks, "tokens": total_tokens}, f)

def process_structure(out_file, seed=42):
    """独立进程：处理 3Di + SS 结构数据"""
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    count = 0
    tokens = 0

    with open(out_file, "wb") as fout:
        # 3Di
        current_seq = []
        with open(f"{DATA_DIR}/pdb_3di.fasta", "r") as fin:
            for line in fin:
                if line.startswith(">"):
                    if current_seq:
                        seq = "".join(current_seq)
                        if len(seq) > 10:
                            text = f"<SEQ_3Di>{seq}</SEQ_3Di>"
                            ids = tokenizer.encode(text, add_special_tokens=False)
                            for j in range(0, len(ids), MAX_LENGTH):
                                chunk = ids[j:j + MAX_LENGTH]
                                if len(chunk) >= MIN_LENGTH:
                                    arr = np.array(chunk, dtype=np.uint32)
                                    fout.write(struct.pack("I", len(chunk)))
                                    fout.write(arr.tobytes())
                                    count += 1
                                    tokens += len(chunk)
                    current_seq = []
                else:
                    current_seq.append(line.strip())

        # SS
        is_secstr = False
        current_ss = []
        with open(f"{DATA_DIR}/ss.txt", "r") as fin:
            for line in fin:
                if line.startswith(">"):
                    if is_secstr and current_ss:
                        ss_seq = "".join(current_ss).replace(" ", "C").strip()
                        if len(ss_seq) > 10:
                            text = f"<SEQ_2D>{ss_seq}</SEQ_2D>"
                            ids = tokenizer.encode(text, add_special_tokens=False)
                            for j in range(0, len(ids), MAX_LENGTH):
                                chunk = ids[j:j + MAX_LENGTH]
                                if len(chunk) >= MIN_LENGTH:
                                    arr = np.array(chunk, dtype=np.uint32)
                                    fout.write(struct.pack("I", len(chunk)))
                                    fout.write(arr.tobytes())
                                    count += 1
                                    tokens += len(chunk)
                    is_secstr = ":secstr" in line
                    current_ss = []
                elif is_secstr:
                    current_ss.append(line.rstrip("\n"))

    print(f"  [Structure] DONE: {count} chunks, {tokens:,} tokens", flush=True)
    with open(out_file + ".stats", "w") as f:
        json.dump({"label": "Structure", "chunks": count, "tokens": tokens}, f)

def process_replay(out_file, seed=42):
    """独立进程：instruction replay"""
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    count = 0
    tokens = 0

    with open(REPLAY_FILE, "r") as fin, open(out_file, "wb") as fout:
        for line in fin:
            if random.random() > 0.25:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if not msgs:
                continue
            parts = []
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "user":
                    parts.append(f"<User>\n{content}\n")
                elif role == "assistant":
                    parts.append(f"<Assistant>\n{content}\n")
            text = "\n".join(parts)
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            for j in range(0, len(ids), MAX_LENGTH):
                chunk = ids[j:j + MAX_LENGTH]
                if len(chunk) >= MIN_LENGTH:
                    arr = np.array(chunk, dtype=np.uint32)
                    fout.write(struct.pack("I", len(chunk)))
                    fout.write(arr.tobytes())
                    count += 1
                    tokens += len(chunk)

    print(f"  [InstructReplay] DONE: {count} chunks, {tokens:,} tokens", flush=True)
    with open(out_file + ".stats", "w") as f:
        json.dump({"label": "InstructionReplay", "chunks": count, "tokens": tokens}, f)

# ================= 主流程 =================
if __name__ == "__main__":
    print("CPT v2 Data Preparation (multiprocess)", flush=True)

    # 定义各数据源的任务
    tasks = [
        ("DNA", f"{DATA_DIR}/dna_32g.txt", 5*1024**3, 0.16, f"{OUT_DIR}/part_dna.bin", 42),
        ("Protein-UniProt", f"{DATA_DIR}/protein_uni_16.txt", int(2.5*1024**3), 0.16, f"{OUT_DIR}/part_prot1.bin", 43),
        ("Protein-LucaOne", f"{DATA_DIR}/protein_lucaone_15g.txt", int(2.5*1024**3), 0.17, f"{OUT_DIR}/part_prot2.bin", 44),
        ("OpenWebText", f"{DATA_DIR}/openwebtext.txt", 5*1024**3, 0.14, f"{OUT_DIR}/part_owt.bin", 45),
    ]

    # S2ORC
    if os.path.exists(LIT_FILE):
        tasks.append(("S2ORC-Biology", LIT_FILE, 10*1024**3, 0.36, f"{OUT_DIR}/part_s2orc.bin", 46))

    # 启动所有文本数据源进程
    procs = []
    for label, filepath, target, rate, outf, seed in tasks:
        print(f"Starting [{label}]...", flush=True)
        p = Process(target=stream_tokenize_file, args=(filepath, target, rate, label, outf, seed))
        p.start()
        procs.append(p)

    # 结构数据进程
    print("Starting [Structure]...", flush=True)
    p_struct = Process(target=process_structure, args=(f"{OUT_DIR}/part_struct.bin", 47))
    p_struct.start()
    procs.append(p_struct)

    # Instruction replay 进程
    if os.path.exists(REPLAY_FILE):
        print("Starting [InstructReplay]...", flush=True)
        p_replay = Process(target=process_replay, args=(f"{OUT_DIR}/part_replay.bin", 48))
        p_replay.start()
        procs.append(p_replay)

    # 等待所有进程完成
    for p in procs:
        p.join()

    print("\nAll processes done. Merging...", flush=True)

    # ================= 合并所有 part_*.bin =================
    import glob
    part_files = sorted(glob.glob(f"{OUT_DIR}/part_*.bin"))
    merged_bin = f"{OUT_DIR}/cpt_all.bin"

    with open(merged_bin, "wb") as fout:
        for pf in part_files:
            if pf.endswith(".stats"):
                continue
            print(f"  Merging {pf} ({os.path.getsize(pf)/1024**3:.2f}GB)", flush=True)
            with open(pf, "rb") as fin:
                while True:
                    data = fin.read(64 * 1024 * 1024)  # 64MB chunks
                    if not data:
                        break
                    fout.write(data)

    # ================= 建索引 =================
    print("Building index...", flush=True)
    index = []
    offset = 0
    with open(merged_bin, "rb") as f:
        while True:
            header = f.read(4)
            if not header or len(header) < 4:
                break
            length = struct.unpack("I", header)[0]
            index.append((offset, length))
            offset += 4 + length * 4
            f.seek(offset)

    random.seed(SEED)
    random.shuffle(index)
    np.save(f"{OUT_DIR}/cpt_index.npy", np.array(index, dtype=np.int64))

    # ================= 汇总 =================
    stats = {}
    for sf in glob.glob(f"{OUT_DIR}/part_*.bin.stats"):
        with open(sf) as f:
            s = json.load(f)
            stats[s["label"]] = s

    total_chunks = sum(s["chunks"] for s in stats.values())
    total_tokens = sum(s["tokens"] for s in stats.values())

    print(f"\n{'='*50}")
    print(f"CPT v2 Data Preparation Complete")
    print(f"{'='*50}")
    for name, s in stats.items():
        pct = s["tokens"] / total_tokens * 100 if total_tokens else 0
        print(f"  {name:20s}: {s['chunks']:>8,} chunks, {s['tokens']:>12,} tokens ({pct:.1f}%)")
    print(f"  {'TOTAL':20s}: {total_chunks:>8,} chunks, {total_tokens:>12,} tokens")
    print(f"\n  Binary: {merged_bin} ({os.path.getsize(merged_bin)/1024**3:.2f} GB)")
    print(f"  Index:  {OUT_DIR}/cpt_index.npy")

    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone!", flush=True)
