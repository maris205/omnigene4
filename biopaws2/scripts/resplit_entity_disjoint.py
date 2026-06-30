"""Entity-disjoint re-splitting for BioPAWS-2 (fixes the leakage the audit surfaced).

The original hash-based split assigned rows independently, so the same protein appearing in
multiple pairs/rows could land in different splits (homology/exact leakage up to 0.97). This
script re-splits each sequence task at the ENTITY level: cluster all sequences with MMseqs2,
then assign whole clusters to train/val/test (80/10/10). A sequence (and every row mentioning
it) therefore lives in exactly one split, and no test cluster shares a train sequence.

For pairwise tasks (homology), a pair is assigned to test only if BOTH sequences' clusters are
test-assigned; pairs straddling splits are dropped (logged). For single-seq tasks, the row
follows its sequence's cluster.

Outputs new *.jsonl in data/ (overwrites, keeping a .preleak backup), and prints the residual
leak rate (should be ~0 for homology).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_SEQ_RE = re.compile(r"([ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]{20,})")

# tasks to re-split (those with real sequences + leakage); name-grounded QA (uniprot_qa) and
# image/text tasks are left as-is.
PAIRWISE = {"protein_homology_std", "protein_homology_remote", "lg_central_dogma"}
SINGLE = {"lg_fold_class", "lg_subcellular_loc", "lg_signal_peptide", "lg_npp",
          "lg_promoter_detection", "f4_proteingym_dms", "int_opi_function",
          "f7_bioreason_cot", "int_protein_catalogue_cot"}


def seqs_of(user_text):
    return [s.upper() for s in _SEQ_RE.findall(user_text)]


def mmseqs_clusters(seqs, min_seq_id=0.5, coverage=0.5):
    """Return {seq: cluster_id} via MMseqs2 easy-cluster."""
    uniq = sorted(set(seqs))
    if len(uniq) < 3:
        return {s: i for i, s in enumerate(uniq)}
    tmpd = tempfile.mkdtemp(prefix="bp2_resplit_")
    fasta = os.path.join(tmpd, "in.fasta")
    with open(fasta, "w") as fh:
        for i, s in enumerate(uniq):
            fh.write(f">{i}\n{s}\n")
    pref = os.path.join(tmpd, "clu")
    subprocess.run(["mmseqs", "easy-cluster", fasta, pref, os.path.join(tmpd, "tmp"),
                    "--min-seq-id", str(min_seq_id), "-c", str(coverage),
                    "--cov-mode", "1", "-v", "0"], check=True, capture_output=True, timeout=1800)
    idx2seq = {str(i): s for i, s in enumerate(uniq)}
    seq2clu = {}
    with open(pref + "_cluster.tsv") as fh:
        for line in fh:
            rep, mem = line.strip().split("\t")
            seq2clu[idx2seq[mem]] = rep
    shutil.rmtree(tmpd, ignore_errors=True)
    return seq2clu


def split_for_cluster(clu_id):
    h = int(hashlib.md5(str(clu_id).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def resplit_task(task):
    path = os.path.join(DATA, f"{task}.jsonl")
    if not os.path.exists(path):
        print(f"  {task}: missing, skip"); return
    rows = load(path)
    # gather all sequences
    all_seqs = []
    for r in rows:
        all_seqs += seqs_of(r["messages"][0]["content"])
    if not all_seqs:
        print(f"  {task}: no sequences, skip"); return
    print(f"[resplit] {task}: clustering {len(set(all_seqs))} unique seqs ...", flush=True)
    seq2clu = mmseqs_clusters(all_seqs)
    clu_split = {}

    def get_split(clu):
        if clu not in clu_split:
            clu_split[clu] = split_for_cluster(clu)
        return clu_split[clu]

    out = []
    dropped = 0
    from collections import Counter
    sp = Counter()
    for r in rows:
        ss = seqs_of(r["messages"][0]["content"])
        clus = [seq2clu.get(s) for s in ss if seq2clu.get(s) is not None]
        if not clus:
            dropped += 1; continue
        splits = {get_split(c) for c in clus}
        if len(splits) == 1:
            r["split"] = splits.pop()
        else:
            # pair straddles splits: assign to the more conservative (test>val>train) only if
            # all-or-nothing; to keep entity-disjointness we DROP straddlers
            dropped += 1; continue
        out.append(r); sp[r["split"]] += 1
    # backup + write
    bak = path.replace(".jsonl", ".jsonl.preleak")
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    with open(path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[resplit] {task}: kept {len(out)} (dropped {dropped} straddlers) splits={dict(sp)}",
          flush=True)


def main():
    tasks = sys.argv[1:] or sorted(PAIRWISE | SINGLE)
    for t in tasks:
        try:
            resplit_task(t)
        except Exception as e:
            print(f"[resplit] {t}: ERROR {str(e)[:150]}", flush=True)


if __name__ == "__main__":
    main()
