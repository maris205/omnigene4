#!/usr/bin/env python
# coding: utf-8
"""
23-three_way_compare.py
三方对比: baseline (Gemma-Instruct) vs CPT-only (0.6 epoch) vs v3 (CPT+SFT+remote)
目的: 拆解 CPT 阶段 vs SFT 阶段各贡献了多少专家分化.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/root/autodl-tmp/dnagpt/outputs/moe_analysis")
NUM_EXPERTS = 128
NUM_LAYERS = 30
TAGS = ["baseline", "cpt", "v3"]
LABELS = {"baseline": "Gemma-4-Instruct", "cpt": "OmniGene-4 CPT (0.6 ep)", "v3": "OmniGene-4 v3 (CPT+SFT+remote)"}
COLORS = {"baseline": "steelblue", "cpt": "orange", "v3": "crimson"}


def load_tag(tag):
    d = np.load(OUT_DIR / f"moe_counts_{tag}.npz")
    tasks = sorted({k.split("__")[0] for k in d.files})
    return {t: d[f"{t}__counts"].astype(np.float64) for t in tasks}, tasks


def normalize_per_layer(c):
    s = c.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return c / s


def js_div(p, q, eps=1e-10):
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def per_layer_pairwise_js(d, tasks):
    pairs = [(i, j) for i in range(len(tasks)) for j in range(i+1, len(tasks))]
    out = np.zeros(NUM_LAYERS)
    for L in range(NUM_LAYERS):
        out[L] = np.mean([js_div(d[tasks[i]][L], d[tasks[j]][L]) for i, j in pairs])
    return out


# Load all three tags
data = {}
for tag in TAGS:
    d, tasks = load_tag(tag)
    data[tag] = {t: normalize_per_layer(d[t]) for t in tasks}
print("tasks:", tasks)

# ============== 1. Per-layer JS (3 curves) ==============
js_all = {tag: per_layer_pairwise_js(data[tag], tasks) for tag in TAGS}

fig, ax = plt.subplots(figsize=(12, 5))
for tag in TAGS:
    ax.plot(range(NUM_LAYERS), js_all[tag], 'o-', label=LABELS[tag],
        color=COLORS[tag], linewidth=1.8, markersize=4)
ax.set_xlabel('Layer index')
ax.set_ylabel('Mean pair-wise JS divergence')
ax.set_title('Cross-task expert differentiation: baseline → CPT → SFT+remote')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "js_three_way.png", dpi=120)
plt.close()
print(f"  wrote {OUT_DIR / 'js_three_way.png'}")

# ============== 2. Decomposition: CPT contribution vs SFT contribution ==============
cpt_gain = js_all["cpt"] - js_all["baseline"]        # 0.6 ep CPT 带来的
sft_gain = js_all["v3"] - js_all["cpt"]              # 再加 SFT+remote 带来的
total = js_all["v3"] - js_all["baseline"]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(NUM_LAYERS)
ax.bar(x - 0.2, cpt_gain, width=0.38, label='CPT contribution', color='orange', alpha=0.9)
ax.bar(x + 0.2, sft_gain, width=0.38, label='SFT+remote contribution', color='crimson', alpha=0.9)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('Layer index')
ax.set_ylabel('Δ mean pair-wise JS divergence')
ax.set_title('Decomposition of differentiation gain: which stage contributed where?')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR / "js_decomposition.png", dpi=120)
plt.close()
print(f"  wrote {OUT_DIR / 'js_decomposition.png'}")

# ============== 3. Per-task JS gain (baseline -> cpt -> v3) ==============
def per_task_js_vs_all(d, tasks, target_task):
    """target_task 与其他任务的平均 JS."""
    ti = tasks.index(target_task)
    avg_per_layer = np.zeros(NUM_LAYERS)
    for L in range(NUM_LAYERS):
        others = [js_div(d[tasks[ti]][L], d[tasks[j]][L]) for j in range(len(tasks)) if j != ti]
        avg_per_layer[L] = np.mean(others)
    return avg_per_layer.mean()

rows = []
for t in tasks:
    bl = per_task_js_vs_all(data["baseline"], tasks, t)
    ct = per_task_js_vs_all(data["cpt"], tasks, t)
    v3 = per_task_js_vs_all(data["v3"], tasks, t)
    rows.append((t, bl, ct, v3))

# 柱状图
fig, ax = plt.subplots(figsize=(11, 5))
task_names = [r[0] for r in rows]
bl_vals = [r[1] for r in rows]
ct_vals = [r[2] for r in rows]
v3_vals = [r[3] for r in rows]
x = np.arange(len(tasks))
w = 0.27
ax.bar(x - w, bl_vals, w, label='baseline', color=COLORS["baseline"])
ax.bar(x,     ct_vals, w, label='CPT only', color=COLORS["cpt"])
ax.bar(x + w, v3_vals, w, label='CPT+SFT', color=COLORS["v3"])
ax.set_xticks(x); ax.set_xticklabels(task_names, rotation=20)
ax.set_ylabel('Avg JS to other tasks (all layers)')
ax.set_title('Per-task isolation from other tasks: did training make this task "stand alone"?')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR / "per_task_gain.png", dpi=120)
plt.close()
print(f"  wrote {OUT_DIR / 'per_task_gain.png'}")

# ============== Summary ==============
print("\n=== Avg pair-wise JS (all layers) ===")
for tag in TAGS:
    print(f"  {tag:<10s}: {js_all[tag].mean():.4f}")
print(f"  Δ baseline->CPT  : {(js_all['cpt']-js_all['baseline']).mean():+.4f}")
print(f"  Δ CPT->SFT+remote: {(js_all['v3']-js_all['cpt']).mean():+.4f}")
print(f"  Δ total          : {(js_all['v3']-js_all['baseline']).mean():+.4f}")

print("\n=== Per-task avg JS-to-others ===")
print(f"  {'task':<14s} {'baseline':>10s} {'CPT':>10s} {'v3':>10s} {'Δ CPT':>9s} {'Δ SFT':>9s}")
for t, bl, ct, v3 in rows:
    print(f"  {t:<14s} {bl:>10.4f} {ct:>10.4f} {v3:>10.4f} "
          f"{ct-bl:>+9.4f} {v3-ct:>+9.4f}")

print("\n=== Top-5 layers for each stage's gain ===")
print("  CPT-stage gain top-5:")
for L in np.argsort(-cpt_gain)[:5]:
    print(f"    Layer {L}: +{cpt_gain[L]:.4f}")
print("  SFT-stage gain top-5:")
for L in np.argsort(-sft_gain)[:5]:
    print(f"    Layer {L}: +{sft_gain[L]:.4f}")

report = {
    "avg_js": {tag: float(js_all[tag].mean()) for tag in TAGS},
    "decomposition": {
        "cpt_contribution": float((js_all["cpt"] - js_all["baseline"]).mean()),
        "sft_contribution": float((js_all["v3"] - js_all["cpt"]).mean()),
        "total": float((js_all["v3"] - js_all["baseline"]).mean()),
    },
    "per_task_avg_js_to_others": {t: {"baseline": bl, "cpt": ct, "v3": v3}
                                   for t, bl, ct, v3 in rows},
    "cpt_gain_per_layer": cpt_gain.tolist(),
    "sft_gain_per_layer": sft_gain.tolist(),
}
with (OUT_DIR / "three_way_report.json").open("w") as f:
    json.dump(report, f, indent=2)
print(f"\nwrote {OUT_DIR / 'three_way_report.json'}")
