#!/usr/bin/env python
"""
F3 (training-stage progression) + F4 (8-panel qualitative)
for the merged OmniGene-4 + MM manuscript.

F3: 3 sub-panels (homology / vision / multi-task gen) across stages
F4: 8 representative examples from qualitative_demo.json
"""
import os, json, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.image import imread
import numpy as np

OUT = "/root/autodl-tmp/dnagpt/outputs/figures_omnigene_mm"
DEMO_JSON = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3/qualitative_demo.json"
CHEBI_BASE = "/root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20"

C_BG = "#faf9f5"; C_TEXT = "#141413"
C_GRAY1 = "#e8e6dc"; C_GRAY2 = "#b0aea5"
C_BLUE = "#3a7fc2"; C_ORANGE = "#d97757"; C_GREEN = "#788c5d"

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                     "savefig.bbox": "tight", "savefig.facecolor": C_BG})


# ================= F3: stage progression =================
stages = ["v5\n(text only)", "Stage 1\n(vision warmup)",
          "Stage 2\n(mixed)", "Stage 3 v2\n(LR 5e-6,\nemb trainable)",
          "Stage 3 v3\n(LR 2e-5,\nemb frozen)"]

# (stage labels above) match these series:
homology_std    = [0.994, np.nan, 0.590, 0.595, 0.850]
homology_remote = [0.826, np.nan, 0.565, 0.600, 0.695]
struct_recog    = [np.nan, 1.00, 1.00, 1.00, 1.00]
struct_cap      = [np.nan, 0.90, 0.88, 0.90, 0.96]
gen_desp        = [np.nan, 0.10, 0.14, 0.16, 0.14]
mt_cell  = [np.nan, np.nan, 0.253, 0.747, 0.953]
mt_mol   = [np.nan, np.nan, 0.320, 0.787, 0.907]
mt_prot  = [np.nan, np.nan, 1.000, 1.000, 1.000]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                         gridspec_kw={"wspace": 0.32})

# Panel A: homology
ax = axes[0]; ax.set_facecolor(C_BG)
x = np.arange(len(stages))
ax.plot(x, homology_std,    "o-", color=C_BLUE,   lw=2.0, ms=8,
        label="standard homology", markeredgecolor=C_BLUE, markerfacecolor="white", mew=2)
ax.plot(x, homology_remote, "s-", color=C_ORANGE, lw=2.0, ms=8,
        label="remote homology",   markeredgecolor=C_ORANGE, markerfacecolor="white", mew=2)
ax.axhline(0.994, color=C_BLUE,   ls=":", lw=0.8, alpha=0.5)
ax.axhline(0.826, color=C_ORANGE, ls=":", lw=0.8, alpha=0.5)
ax.text(4.2, 0.994, "v5", fontsize=7, color=C_BLUE, va="center")
ax.text(4.2, 0.826, "v5", fontsize=7, color=C_ORANGE, va="center")
for xi, yi in zip(x, homology_std):
    if not np.isnan(yi):
        ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7, color=C_BLUE)
for xi, yi in zip(x, homology_remote):
    if not np.isnan(yi):
        ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7, color=C_ORANGE)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=7.5)
ax.set_ylim(0.4, 1.05); ax.set_ylabel("Accuracy", fontsize=10)
ax.set_title("(a) Protein homology recovery", fontsize=11, weight="bold")
ax.grid(alpha=0.25, ls=":")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

# Panel B: vision
ax = axes[1]; ax.set_facecolor(C_BG)
ax.plot(x, struct_recog, "o-", color=C_GREEN,  lw=2.0, ms=8,
        label="struct_recog",      markeredgecolor=C_GREEN,  markerfacecolor="white", mew=2)
ax.plot(x, struct_cap,   "s-", color=C_BLUE,   lw=2.0, ms=8,
        label="struct_cap",        markeredgecolor=C_BLUE,   markerfacecolor="white", mew=2)
ax.plot(x, gen_desp,     "^-", color=C_ORANGE, lw=2.0, ms=8,
        label="general_desp",      markeredgecolor=C_ORANGE, markerfacecolor="white", mew=2)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=7.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy", fontsize=10)
ax.set_title("(b) Vision (Vis-CheBI20)", fontsize=11, weight="bold")
ax.grid(alpha=0.25, ls=":")
ax.legend(loc="center right", fontsize=8, framealpha=0.9)

# Panel C: multi-task gen
ax = axes[2]; ax.set_facecolor(C_BG)
ax.plot(x, mt_cell, "o-", color=C_GREEN,  lw=2.0, ms=8,
        label="Cell ID",      markeredgecolor=C_GREEN,  markerfacecolor="white", mew=2)
ax.plot(x, mt_mol,  "s-", color=C_BLUE,   lw=2.0, ms=8,
        label="Mol descriptor", markeredgecolor=C_BLUE,   markerfacecolor="white", mew=2)
ax.plot(x, mt_prot, "^-", color=C_ORANGE, lw=2.0, ms=8,
        label="Protein homology", markeredgecolor=C_ORANGE, markerfacecolor="white", mew=2)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=7.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("Keyword score", fontsize=10)
ax.set_title("(c) Multi-task generation", fontsize=11, weight="bold")
ax.grid(alpha=0.25, ls=":")
ax.legend(loc="center right", fontsize=8, framealpha=0.9)

fig.suptitle("Figure 3.  Capability progression across training stages of OmniGene-4-MM",
             fontsize=12.5, weight="bold", y=1.02)
plt.savefig(f"{OUT}/fig3_stage_progression.pdf")
plt.savefig(f"{OUT}/fig3_stage_progression.png", dpi=200)
plt.close()
print("wrote fig3_stage_progression")


# ================= F4: 8-panel qualitative =================
with open(DEMO_JSON) as f:
    demo = json.load(f)

# Index helpers
v_by_task = {}
for r in demo["vision"]:
    v_by_task.setdefault(r["task"], []).append(r)
mt_by_cat = {}
for r in demo["multi_task"]:
    mt_by_cat.setdefault(r["category"], []).append(r)

# Curate the 8 best illustrative items
panels = [
    # vision panels
    {"kind":"vision", "data": v_by_task["struct_cap"][0],   "panel":"A",
     "title":"struct_cap — list functional groups"},
    {"kind":"vision", "data": v_by_task["general_desp"][0], "panel":"B",
     "title":"general_desp — describe the molecule"},
    {"kind":"vision", "data": v_by_task["trans_iupac"][1],  "panel":"C",
     "title":"trans_iupac — IUPAC naming"},
    {"kind":"vision", "data": v_by_task["struct_recog"][1], "panel":"D",
     "title":"struct_recog — highlighted group"},
    # text panels
    {"kind":"homology", "data": demo["homology"][1],  "panel":"E",
     "title":"standard homology — positive pair"},
    {"kind":"homology", "data": demo["homology"][5],  "panel":"F",
     "title":"remote homology — positive pair"},
    {"kind":"text",     "data": mt_by_cat["Cell"][0], "panel":"G",
     "title":"cell type from markers"},
    {"kind":"text",     "data": mt_by_cat["Mol"][1],  "panel":"H",
     "title":"physico-chemical descriptor from SMILES"},
]

def wrap(s, width):
    s = (s or "").strip().replace("**", "")
    s = s.replace("\n\n", "\n").replace("$-\\text{", "-").replace("}$", "")
    s = s.replace("$\\text{", "").replace("$", "").replace("\\", "")
    paras = s.split("\n")
    out = []
    for p in paras:
        out.extend(textwrap.wrap(p, width=width) or [""])
    return "\n".join(out)

def trunc_lines(s, max_lines):
    lines = s.split("\n")
    if len(lines) <= max_lines: return s
    return "\n".join(lines[:max_lines-1] + [lines[max_lines-1][:60].rstrip() + "…"])

fig = plt.figure(figsize=(15, 16))
fig.patch.set_facecolor(C_BG)
gs = fig.add_gridspec(4, 2, hspace=0.18, wspace=0.10, left=0.025,
                       right=0.985, top=0.96, bottom=0.02)

for i, p in enumerate(panels):
    ax = fig.add_subplot(gs[i//2, i%2])
    ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # outer border
    ax.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        fc="white", ec=C_GRAY2, lw=0.7))
    # header bar
    ax.add_patch(FancyBboxPatch((0.005, 0.91), 0.99, 0.085,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        fc=C_BLUE, ec=C_BLUE))
    ax.text(0.025, 0.952, p["panel"], ha="left", va="center",
            fontsize=15, weight="bold", color="white")
    ax.text(0.075, 0.952, p["title"], ha="left", va="center",
            fontsize=10.5, color="white")

    if p["kind"] == "vision":
        d = p["data"]
        # Image at left
        img_path = f"{CHEBI_BASE}/{d['image']}"
        if os.path.exists(img_path):
            img = imread(img_path)
            ax_img = ax.inset_axes([0.03, 0.10, 0.30, 0.78])
            ax_img.imshow(img); ax_img.axis("off")
        # Text on right (x_text=0.36 onward)
        xt = 0.36
        ax.text(xt, 0.86, "Prompt", fontsize=8.5, weight="bold", color=C_TEXT)
        ax.text(xt, 0.82, wrap(d["prompt"], 65), fontsize=8, color=C_TEXT, va="top")
        ax.text(xt, 0.74, "Reference", fontsize=8.5, weight="bold", color=C_GRAY2)
        ref = trunc_lines(wrap(d["reference"], 65), 4)
        ax.text(xt, 0.71, ref, fontsize=7.7, color=C_GRAY2, va="top")
        ax.text(xt, 0.46, "v3 prediction", fontsize=8.5, weight="bold", color=C_BLUE)
        pred = trunc_lines(wrap(d["prediction"], 65), 7)
        ax.text(xt, 0.43, pred, fontsize=7.7, color=C_TEXT, va="top")

    elif p["kind"] == "homology":
        d = p["data"]
        ax.text(0.03, 0.86, "Sequence 1", fontsize=8.5, weight="bold")
        ax.text(0.03, 0.81, wrap(d["seq1_preview"], 90), fontsize=7,
                family="monospace", color=C_TEXT, va="top")
        ax.text(0.03, 0.65, "Sequence 2", fontsize=8.5, weight="bold")
        ax.text(0.03, 0.60, wrap(d["seq2_preview"], 90), fontsize=7,
                family="monospace", color=C_TEXT, va="top")
        ax.text(0.03, 0.42, "Gold label", fontsize=8.5, weight="bold", color=C_GRAY2)
        ax.text(0.03, 0.37, "Homologous" if d["label"]==1 else "Non-Homologous",
                fontsize=9, color=C_GRAY2)
        ax.text(0.03, 0.27, "v3 prediction", fontsize=8.5, weight="bold", color=C_BLUE)
        match = (d["prediction"].lower().startswith("homolog") and d["label"]==1) or \
                (d["prediction"].lower().startswith("non") and d["label"]==0)
        col = C_GREEN if match else C_ORANGE
        sym = "✓ correct" if match else "✗ wrong"
        ax.text(0.03, 0.22, d["prediction"], fontsize=10, weight="bold", color=col)
        ax.text(0.03, 0.13, sym, fontsize=9, color=col, weight="bold")

    elif p["kind"] == "text":
        d = p["data"]
        ax.text(0.03, 0.86, "Instruction", fontsize=8.5, weight="bold")
        ax.text(0.03, 0.82, wrap(d["instruction"], 95), fontsize=8,
                color=C_TEXT, va="top")
        y_cur = 0.70
        if d.get("input"):
            ax.text(0.03, y_cur, "Input", fontsize=8.5, weight="bold")
            ax.text(0.03, y_cur-0.04, wrap(d["input"], 95), fontsize=7.7,
                    family="monospace", color=C_TEXT, va="top")
            y_cur -= 0.18
        ax.text(0.03, y_cur, "Reference", fontsize=8.5, weight="bold", color=C_GRAY2)
        ax.text(0.03, y_cur-0.04, trunc_lines(wrap(d["reference"], 95), 4),
                fontsize=7.7, color=C_GRAY2, va="top")
        ax.text(0.03, y_cur-0.26, "v3 prediction", fontsize=8.5, weight="bold", color=C_BLUE)
        ax.text(0.03, y_cur-0.30, trunc_lines(wrap(d["prediction"], 95), 4),
                fontsize=7.7, color=C_TEXT, va="top")

fig.suptitle("Figure 4.  Qualitative showcase across vision, homology, and multi-task generation",
             fontsize=13, weight="bold", y=0.985)
plt.savefig(f"{OUT}/fig4_qualitative.pdf")
plt.savefig(f"{OUT}/fig4_qualitative.png", dpi=180)
plt.close()
print("wrote fig4_qualitative")
print(f"all in {OUT}")
