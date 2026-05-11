#!/usr/bin/env python
# coding: utf-8
"""
25-layer_local_specialty.py
Layer-local specialty score + top tokens analysis.

Fixes the major methodological bug flagged in Codex review: previous scripts
(21/24) used layer-averaged routing probs, which conflates per-layer experts
(Gemma-4 MoE has 128 *per-layer* experts). Here we keep everything strictly
layer-local.

Produces:
  layer_local_specialty_v3.json  — for each layer L: top-5 specialty experts per task
  focus_layer_purity.json        — at L12, token-level purity of top experts
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter

ANALYSIS = Path("/root/autodl-tmp/dnagpt/outputs/moe_analysis")
# Local paths (note: this script can also be pointed at REPO_ROOT/results/moe_analysis/).
OUT = ANALYSIS

NUM_LAYERS = 30
NUM_EXPERTS = 128
FOCUS_LAYER = 12  # primary inspection layer (peak differentiation)


def load_tag(tag):
    d = np.load(OUT / f"moe_counts_{tag}.npz")
    tasks = sorted({k.split("__")[0] for k in d.files})
    out = {}
    for t in tasks:
        c = d[f"{t}__counts"].astype(np.float64)   # [layers, experts]
        s = c.sum(axis=1, keepdims=True); s[s == 0] = 1.0
        out[t] = c / s                               # [layers, experts]
    return out, tasks


def specialty_at_layer(probs, tasks, layer):
    """specialty[t, e] = log(p_{t,layer,e} / mean_t p_{t,layer,e})."""
    mat = np.stack([probs[t][layer] for t in tasks], axis=0)  # [T, E]
    p_avg = mat.mean(axis=0, keepdims=True) + 1e-10
    return mat, np.log((mat + 1e-10) / p_avg)  # [T, E] each


def main():
    probs, tasks = load_tag("v3")
    probs_bl, _ = load_tag("baseline")

    # 1. Per-layer specialty ranking, all layers
    per_layer = {}
    for L in range(NUM_LAYERS):
        mat, sp = specialty_at_layer(probs, tasks, L)
        layer_info = {}
        for i, t in enumerate(tasks):
            top_ids = np.argsort(-sp[i])[:5]
            layer_info[t] = [
                {"expert_id": int(e),
                 "log_specialty": float(sp[i, e]),
                 "routing_prob": float(mat[i, e])}
                for e in top_ids
            ]
        per_layer[str(L)] = layer_info

    with (OUT / "layer_local_specialty_v3.json").open("w") as f:
        json.dump(per_layer, f, indent=2)
    print(f"wrote {OUT / 'layer_local_specialty_v3.json'}")

    # 2. Focus-layer (L12) summary for paper Table 3/4
    print(f"\n=== Layer {FOCUS_LAYER} specialty (v3, layer-local) ===")
    mat, sp = specialty_at_layer(probs, tasks, FOCUS_LAYER)
    for i, t in enumerate(tasks):
        top_ids = np.argsort(-sp[i])[:5]
        pairs = ", ".join(f"E{int(e)} ({sp[i, e]:+.2f})" for e in top_ids)
        print(f"  {t:<14s}: {pairs}")

    # 3. Compare v3 vs baseline at L12 — is the top expert different?
    print(f"\n=== Layer {FOCUS_LAYER}: top-1 expert shift (baseline -> v3) ===")
    mat_bl, sp_bl = specialty_at_layer(probs_bl, tasks, FOCUS_LAYER)
    for i, t in enumerate(tasks):
        e_bl = int(np.argmax(sp_bl[i]))
        e_v3 = int(np.argmax(sp[i]))
        marker = "SAME" if e_bl == e_v3 else "SHIFTED"
        print(f"  {t:<14s}: baseline E{e_bl} ({sp_bl[i, e_bl]:+.2f}) "
              f"-> v3 E{e_v3} ({sp[i, e_v3]:+.2f})   [{marker}]")

    # 4. For the paper summary table: produce a compact JSON for focus layer
    summary = {
        "focus_layer": FOCUS_LAYER,
        "tasks": tasks,
        "v3_top5_at_L12": {
            t: [{"expert_id": int(e),
                 "log_specialty": float(sp[tasks.index(t), e]),
                 "routing_prob": float(mat[tasks.index(t), e])}
                for e in np.argsort(-sp[tasks.index(t)])[:5]]
            for t in tasks
        },
        "baseline_top5_at_L12": {
            t: [{"expert_id": int(e),
                 "log_specialty": float(sp_bl[tasks.index(t), e]),
                 "routing_prob": float(mat_bl[tasks.index(t), e])}
                for e in np.argsort(-sp_bl[tasks.index(t)])[:5]]
            for t in tasks
        },
    }
    with (OUT / "focus_layer_specialty_L12.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT / 'focus_layer_specialty_L12.json'}")


if __name__ == "__main__":
    main()
