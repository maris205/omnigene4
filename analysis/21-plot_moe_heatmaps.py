#!/usr/bin/env python
# coding: utf-8
"""
21-plot_moe_heatmaps.py
分析 v3 vs baseline 的 MoE 专家激活模式.
产出:
- 每个模型的 task × expert 热图 (所有层求和)
- v3 与 baseline 的 delta 热图 (训练引起的专家偏移)
- 任务两两 JS 散度矩阵 (v3 vs baseline 对比)
- Top-N 专家的任务归属表
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUT_DIR = Path("/root/autodl-tmp/dnagpt/outputs/moe_analysis")
NUM_EXPERTS = 128
NUM_LAYERS = 30


def load_tag(tag):
    p = OUT_DIR / f"moe_counts_{tag}.npz"
    d = np.load(p)
    tasks = sorted({k.split("__")[0] for k in d.files})
    out = {}
    for t in tasks:
        c = d[f"{t}__counts"]           # [layers, experts]
        n = int(d[f"{t}__tokens"][0])
        out[t] = {"counts": c, "tokens": n}
    return out


def normalize(d):
    """对每层每任务归一化为概率分布 (expert 选中比例)."""
    out = {}
    for t, v in d.items():
        c = v["counts"].astype(np.float64)
        s = c.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        out[t] = c / s              # [layers, experts]
    return out


def task_expert_matrix(d_norm, tasks):
    """合并所有层 (求平均) 得到 task × expert."""
    mat = np.stack([d_norm[t].mean(axis=0) for t in tasks], axis=0)  # [tasks, experts]
    return mat


def js_divergence(p, q, eps=1e-10):
    """Jensen-Shannon divergence, 值域 [0, log 2]."""
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * np.log(a / b))
    return 0.5 * (kl(p, m) + kl(q, m))


def pairwise_js(mat, tasks):
    """返回 [T, T] JS 矩阵."""
    T = len(tasks)
    J = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            J[i, j] = js_divergence(mat[i], mat[j])
    return J


def specialty_score(mat, tasks):
    """每个 expert 对某 task 的 specialty = log(p_task / p_avg)."""
    p_avg = mat.mean(axis=0, keepdims=True) + 1e-10
    p_t = mat + 1e-10
    return np.log(p_t / p_avg)       # [tasks, experts]


def plot_task_expert_heatmap(mat, tasks, title, path, vmax=None):
    plt.figure(figsize=(16, max(3, 0.5 * len(tasks))))
    im = plt.imshow(mat, aspect='auto', cmap='viridis', vmax=vmax)
    plt.colorbar(im, label='avg routing prob')
    plt.yticks(range(len(tasks)), tasks)
    plt.xlabel('Expert ID (0..127)')
    plt.ylabel('Task')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  wrote {path}")


def plot_delta(mat_v3, mat_bl, tasks, path):
    delta = mat_v3 - mat_bl
    vmax = max(abs(delta.min()), abs(delta.max()))
    plt.figure(figsize=(16, max(3, 0.5 * len(tasks))))
    im = plt.imshow(delta, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label='Δ routing prob (v3 - baseline)')
    plt.yticks(range(len(tasks)), tasks)
    plt.xlabel('Expert ID (0..127)')
    plt.ylabel('Task')
    plt.title('Expert usage shift after bio-training (v3 - Gemma-4-Instruct baseline)')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  wrote {path}")


def plot_js_matrix(J, tasks, title, path):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(J, cmap='magma')
    plt.colorbar(im, label='JS divergence')
    plt.xticks(range(len(tasks)), tasks, rotation=45, ha='right')
    plt.yticks(range(len(tasks)), tasks)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            plt.text(j, i, f"{J[i,j]:.2f}", ha='center', va='center',
                color='w' if J[i,j] < J.max()/2 else 'k', fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  wrote {path}")


def main():
    v3 = load_tag("v3")
    bl = load_tag("baseline")
    tasks = sorted(set(v3.keys()) & set(bl.keys()))
    print("Tasks:", tasks)

    n_v3 = normalize(v3)
    n_bl = normalize(bl)

    mat_v3 = task_expert_matrix(n_v3, tasks)
    mat_bl = task_expert_matrix(n_bl, tasks)

    # 1. 热图
    vmax = max(mat_v3.max(), mat_bl.max())
    plot_task_expert_heatmap(mat_v3, tasks,
        'OmniGene-4 v3 (CPT+SFT+remote): avg expert routing by task',
        OUT_DIR / "heatmap_v3.png", vmax=vmax)
    plot_task_expert_heatmap(mat_bl, tasks,
        'Gemma-4-Instruct baseline: avg expert routing by task',
        OUT_DIR / "heatmap_baseline.png", vmax=vmax)

    # 2. Delta 热图
    plot_delta(mat_v3, mat_bl, tasks, OUT_DIR / "heatmap_delta.png")

    # 3. JS 散度
    J_v3 = pairwise_js(mat_v3, tasks)
    J_bl = pairwise_js(mat_bl, tasks)
    plot_js_matrix(J_v3, tasks,
        'Task-pair JS divergence (OmniGene-4 v3)',
        OUT_DIR / "js_v3.png")
    plot_js_matrix(J_bl, tasks,
        'Task-pair JS divergence (Gemma-4-Instruct baseline)',
        OUT_DIR / "js_baseline.png")
    # Δ JS: 训练让哪些任务更分化了
    delta_js = J_v3 - J_bl
    vmax = max(abs(delta_js.min()), abs(delta_js.max()))
    plt.figure(figsize=(7, 6))
    im = plt.imshow(delta_js, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label='Δ JS (v3 - baseline)')
    plt.xticks(range(len(tasks)), tasks, rotation=45, ha='right')
    plt.yticks(range(len(tasks)), tasks)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            plt.text(j, i, f"{delta_js[i,j]:+.2f}", ha='center', va='center',
                color='k', fontsize=7)
    plt.title('Δ task-pair JS divergence (v3 − baseline)\n+ = more differentiated after training')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "js_delta.png", dpi=120)
    plt.close()
    print(f"  wrote {OUT_DIR / 'js_delta.png'}")

    # 4. Top-N specialty experts (v3)
    sp_v3 = specialty_score(mat_v3, tasks)     # [tasks, experts]
    print("\n=== Top-5 specialty experts per task (v3) ===")
    report = {"tasks": tasks, "top_experts_v3": {}, "top_experts_baseline": {}}
    for i, t in enumerate(tasks):
        top_ids = np.argsort(-sp_v3[i])[:5]
        scores = sp_v3[i, top_ids]
        pairs = [(int(e), float(s), float(mat_v3[i, e])) for e, s in zip(top_ids, scores)]
        report["top_experts_v3"][t] = pairs
        print(f"  {t:<14s}: " + ", ".join(f"E{e}(log_sp={s:+.2f}, p={p:.3f})" for e, s, p in pairs))

    sp_bl = specialty_score(mat_bl, tasks)
    print("\n=== Top-5 specialty experts per task (baseline) ===")
    for i, t in enumerate(tasks):
        top_ids = np.argsort(-sp_bl[i])[:5]
        scores = sp_bl[i, top_ids]
        pairs = [(int(e), float(s), float(mat_bl[i, e])) for e, s in zip(top_ids, scores)]
        report["top_experts_baseline"][t] = pairs
        print(f"  {t:<14s}: " + ", ".join(f"E{e}(log_sp={s:+.2f}, p={p:.3f})" for e, s, p in pairs))

    # 5. 全局统计
    avg_js_v3 = J_v3[np.triu_indices(len(tasks), k=1)].mean()
    avg_js_bl = J_bl[np.triu_indices(len(tasks), k=1)].mean()
    print(f"\n=== Avg pair-wise JS ===")
    print(f"  baseline: {avg_js_bl:.4f}")
    print(f"  v3      : {avg_js_v3:.4f}")
    print(f"  Δ       : {avg_js_v3 - avg_js_bl:+.4f}  "
          f"({'more differentiated' if avg_js_v3 > avg_js_bl else 'less differentiated'})")
    report["avg_js"] = {"v3": float(avg_js_v3), "baseline": float(avg_js_bl),
                        "delta": float(avg_js_v3 - avg_js_bl)}

    with (OUT_DIR / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
