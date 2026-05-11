#!/usr/bin/env python
# coding: utf-8
"""
22-analyze_per_layer.py
按层维度分析 MoE 专家分化:
  1. 每层 routing 熵 (越低 = 越专一) - line plot
  2. 每层任务两两 JS 散度均值 - 训练后哪些层分化最强
  3. 关键层 (浅/中/深) 的 delta 热图 - 5 张
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


def load_tag(tag):
    d = np.load(OUT_DIR / f"moe_counts_{tag}.npz")
    tasks = sorted({k.split("__")[0] for k in d.files})
    return {t: d[f"{t}__counts"] for t in tasks}, tasks


def normalize_per_layer(counts):
    """[layers, experts] counts -> [layers, experts] probs"""
    c = counts.astype(np.float64)
    s = c.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return c / s


def entropy(p, eps=1e-12):
    return -np.sum(p * np.log(p + eps), axis=-1)


def js_div(p, q, eps=1e-10):
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


v3_counts, tasks = load_tag("v3")
bl_counts, _ = load_tag("baseline")
v3 = {t: normalize_per_layer(v3_counts[t]) for t in tasks}
bl = {t: normalize_per_layer(bl_counts[t]) for t in tasks}

H_MAX = np.log(NUM_EXPERTS)
print(f"max possible entropy = log({NUM_EXPERTS}) = {H_MAX:.3f}")

# ============== 1. Per-layer entropy ==============
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
for i, t in enumerate(tasks):
    h_bl = entropy(bl[t])
    h_v3 = entropy(v3[t])
    axes[0].plot(range(NUM_LAYERS), h_bl, color=colors[i], label=t, linewidth=1.6)
    axes[1].plot(range(NUM_LAYERS), h_v3, color=colors[i], label=t, linewidth=1.6)

for ax, title in zip(axes, ['Gemma-4-Instruct baseline', 'OmniGene-4 v3 (CPT+SFT+remote)']):
    ax.axhline(H_MAX, ls=':', c='gray', alpha=0.5, label=f'max entropy log(128)={H_MAX:.2f}')
    ax.set_ylabel('Routing entropy (nats)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_ylim(2.5, H_MAX + 0.1)
axes[1].set_xlabel('Layer index')
axes[0].legend(loc='lower right', ncol=2, fontsize=8)
plt.suptitle('Per-layer routing entropy: lower = experts more specialized')
plt.tight_layout()
plt.savefig(OUT_DIR / "per_layer_entropy.png", dpi=120)
plt.close()
print(f"  wrote {OUT_DIR / 'per_layer_entropy.png'}")

# ============== 2. Per-layer mean pair-wise JS ==============
def per_layer_pairwise_js(d):
    out = np.zeros(NUM_LAYERS)
    pairs = [(i, j) for i in range(len(tasks)) for j in range(i+1, len(tasks))]
    for L in range(NUM_LAYERS):
        vals = [js_div(d[tasks[i]][L], d[tasks[j]][L]) for i, j in pairs]
        out[L] = np.mean(vals)
    return out

js_v3 = per_layer_pairwise_js(v3)
js_bl = per_layer_pairwise_js(bl)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(range(NUM_LAYERS), js_bl, 'o-', label='Gemma-4-Instruct baseline', color='steelblue', linewidth=1.8)
ax.plot(range(NUM_LAYERS), js_v3, 's-', label='OmniGene-4 v3', color='crimson', linewidth=1.8)
ax.fill_between(range(NUM_LAYERS), js_bl, js_v3, where=(js_v3 > js_bl),
    alpha=0.2, color='crimson', label='v3 more differentiated')
ax.fill_between(range(NUM_LAYERS), js_bl, js_v3, where=(js_v3 < js_bl),
    alpha=0.2, color='steelblue', label='baseline more differentiated')
ax.set_xlabel('Layer index'); ax.set_ylabel('Mean pair-wise JS divergence')
ax.set_title('Cross-task expert routing differentiation, by layer')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_layer_js.png", dpi=120)
plt.close()
print(f"  wrote {OUT_DIR / 'per_layer_js.png'}")

# ============== 3. Key-layer delta heatmaps ==============
key_layers = [0, 7, 14, 21, 29]
fig, axes = plt.subplots(len(key_layers), 1, figsize=(16, 2.5 * len(key_layers)), sharex=True)
for ax, L in zip(axes, key_layers):
    delta = np.stack([v3[t][L] - bl[t][L] for t in tasks])
    vmax = max(abs(delta.min()), abs(delta.max()))
    im = ax.imshow(delta, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks)
    ax.set_title(f'Layer {L}')
    plt.colorbar(im, ax=ax, fraction=0.018)
axes[-1].set_xlabel('Expert ID (0..127)')
plt.suptitle('Δ routing prob (v3 − baseline) at key layers (shallow → deep)')
plt.tight_layout()
plt.savefig(OUT_DIR / "delta_key_layers.png", dpi=120)
plt.close()
print(f"  wrote {OUT_DIR / 'delta_key_layers.png'}")

# ============== Numeric summary ==============
mean_h_bl = {t: float(entropy(bl[t]).mean()) for t in tasks}
mean_h_v3 = {t: float(entropy(v3[t]).mean()) for t in tasks}
summary = {
    "max_entropy_uniform": float(H_MAX),
    "mean_entropy_per_task": {
        "baseline": mean_h_bl,
        "v3": mean_h_v3,
        "delta": {t: mean_h_v3[t] - mean_h_bl[t] for t in tasks},
    },
    "mean_pairwise_js_per_layer": {
        "baseline": js_bl.tolist(),
        "v3": js_v3.tolist(),
        "delta_avg_all_layers": float((js_v3 - js_bl).mean()),
        "max_delta_layer": int((js_v3 - js_bl).argmax()),
        "max_delta_value": float((js_v3 - js_bl).max()),
    },
}
with (OUT_DIR / "per_layer_report.json").open("w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Mean entropy per task (lower = more specialized) ===")
print(f"  {'task':<14s} {'baseline':>10s} {'v3':>10s} {'Δ':>10s}")
for t in tasks:
    print(f"  {t:<14s} {mean_h_bl[t]:>10.4f} {mean_h_v3[t]:>10.4f} "
          f"{mean_h_v3[t]-mean_h_bl[t]:>+10.4f}")

print(f"\n=== Per-layer JS, top-5 layers with biggest Δ ===")
delta_js = js_v3 - js_bl
top5 = np.argsort(-delta_js)[:5]
for L in top5:
    print(f"  Layer {L:>2d}: baseline={js_bl[L]:.4f}, v3={js_v3[L]:.4f}, Δ={delta_js[L]:+.4f}")

print(f"\nwrote {OUT_DIR / 'per_layer_report.json'}")
