#!/usr/bin/env python
"""
Two fallback figures for the merged OmniGene-4 + MM manuscript:
  fig1_architecture.pdf/png  -- model architecture
  fig2_differentiation.pdf/png -- vs AIDO.Protein / Tripathi 2025
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

OUT = "/root/autodl-tmp/dnagpt/outputs/figures_omnigene_mm"
os.makedirs(OUT, exist_ok=True)

# Brand-aligned palette (Anthropic-friendly + scientific)
C_BG     = "#faf9f5"
C_TEXT   = "#141413"
C_ACCENT = "#3a7fc2"  # blue (ours)
C_GRAY1  = "#e8e6dc"
C_GRAY2  = "#b0aea5"
C_ORANGE = "#d97757"
C_GREEN  = "#788c5d"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.facecolor": C_BG,
})

# ---------------------------------------------------------------
# FIGURE 1 — Architecture
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
ax.set_facecolor(C_BG)

def box(x, y, w, h, label, fc, ec=C_TEXT, ls="-", fontsize=8.5,
        weight="normal", lw=0.7, ha="center", va="center"):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.18,rounding_size=0.6",
        fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, label, ha=ha, va=va, fontsize=fontsize,
            color=C_TEXT, weight=weight)

def arrow(x1, y1, x2, y2, color=C_TEXT, lw=0.9, style="->"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle=style, color=color, lw=lw,
        mutation_scale=10, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# Title
ax.text(50, 57.5, "OmniGene-4-MM: multi-modal MoE for biological structural discovery",
        ha="center", fontsize=12.5, weight="bold", color=C_TEXT)

# ---- LEFT: inputs ----
ax.text(8, 51.5, "Inputs", fontsize=10, weight="bold", color=C_TEXT, ha="center")
inputs = [
    ("Molecular image\n(Vis-CheBI20)",        C_ORANGE, 47),
    ("Medical / pathology\n(PubMedVision)",   C_ORANGE, 41),
    ("Protein sequence\n(15B residues)",      C_GREEN,  35),
    ("DNA sequence\n(32B bases)",             C_GREEN,  29),
    ("3Di alphabet\n(Foldseek)",              C_GREEN,  23),
    ("Natural language\n(PAWS-X, SFT)",       C_GRAY2,  17),
]
for label, fc, y in inputs:
    box(1, y, 14, 4, label, fc=fc, fontsize=7.5)

# ---- Vision tower ----
box(18, 41, 12, 8.5,
    "Gemma4 Vision Tower\n27 layers, 1152 hidden\npatch 16  →  2520 patches",
    fc="#dbe9f7", lw=1.0, fontsize=8)

arrow(15, 49, 18, 47)
arrow(15, 43, 18, 45)

# ---- Backbone ----
back_x, back_y, back_w, back_h = 33, 14, 36, 36
box(back_x, back_y, back_w, back_h, "", fc="#fff6ed", ec=C_ORANGE, lw=1.4)
ax.text(back_x+back_w/2, back_y+back_h-2.5,
        "Gemma-4 MoE Backbone  (26B-A4B)",
        ha="center", fontsize=10, weight="bold", color=C_TEXT)
ax.text(back_x+back_w/2, back_y+back_h-5,
        "30 layers · 128 experts/layer · top-8 routing",
        ha="center", fontsize=8.2, color=C_TEXT)

# A representative expanded layer
lx, ly, lw_, lh_ = back_x+2.2, back_y+5, back_w-4.4, back_h-13
box(lx, ly, lw_, lh_, "", fc="white", ec=C_GRAY2, lw=0.8)
ax.text(lx+lw_/2, ly+lh_-2, "Layer L  (expanded view)",
        ha="center", fontsize=8.5, weight="bold")

# Self-attention
box(lx+1.5, ly+lh_-7, 11, 4,
    "Self-Attn\nQ K V O",
    fc="#fae0d8", fontsize=7.5)
ax.text(lx+7, ly+lh_-8.4, "LoRA r=64", ha="center",
        fontsize=6.5, color=C_ORANGE, weight="bold")

# Router
box(lx+14.5, ly+lh_-7, 8, 4, "Router", fc="#dbe9f7", fontsize=7.5)
ax.text(lx+18.5, ly+lh_-8.4, "LoRA on router.proj", ha="center",
        fontsize=6.5, color=C_ORANGE, weight="bold")

# Experts grid
ex0, ey0 = lx+1.5, ly+1.5
for i in range(16):
    for j in range(8):
        x = ex0 + i*1.8
        y = ey0 + j*1.0
        is_active = (i+j) % 5 == 0
        ax.add_patch(Rectangle((x, y), 1.5, 0.8,
            fc=(C_ORANGE if is_active else C_GRAY1),
            ec=C_GRAY2, lw=0.3))
ax.text(lx+lw_/2, ly+0.6, "128 experts  ·  top-8 active per token  (LoRA on gate / up / down)",
        ha="center", fontsize=7.3, color=C_TEXT)

# Vision tower → backbone
arrow(30, 45, 33, 35, lw=1.0)
ax.text(31.5, 41, "visual\npatches", fontsize=6.5, color=C_TEXT)

# Sequence → backbone
for y in (35, 29, 23, 17):
    arrow(15, y+2, 33, y+2)

# ---- RIGHT: heads ----
ax.text(85, 51.5, "Output heads", fontsize=10, weight="bold", color=C_TEXT, ha="center")
heads = [
    ("Vision-language\nstructure captions",  46),
    ("Text head\nhomology · cell · SFT",     35),
    ("Sequence head\nprotein · DNA · 3Di",   24),
]
for label, y in heads:
    box(78, y, 16, 6.5, label, fc="#dde8d4", lw=0.9, fontsize=8)
    arrow(69, 32+(y-35)*0.4, 78, y+3.2)

# ---- Training timeline at bottom ----
ax.text(50, 11, "Training pipeline (single H20 GPU)",
        ha="center", fontsize=9.5, weight="bold")
stages = [
    ("Stage 1\nVision warmup",             18, "#dbe9f7"),
    ("Stage 2\nMixed text + vision",       45, "#dbe9f7"),
    ("Stage 3 v3\nHomology specialty\nLR 2e-5, 3000 steps", 73, C_ACCENT),
]
for label, x, fc in stages:
    fontc = "white" if fc == C_ACCENT else C_TEXT
    p = FancyBboxPatch((x-9.5, 3), 19, 6,
        boxstyle="round,pad=0.18,rounding_size=0.6",
        fc=fc, ec=C_TEXT, lw=0.7)
    ax.add_patch(p)
    ax.text(x, 6, label, ha="center", va="center", fontsize=7.8,
            color=fontc, weight=("bold" if fc==C_ACCENT else "normal"))

arrow(28, 6, 36, 6, lw=1.2)
arrow(55, 6, 63, 6, lw=1.2)

plt.savefig(f"{OUT}/fig1_architecture.pdf")
plt.savefig(f"{OUT}/fig1_architecture.png", dpi=200)
plt.close()
print("wrote fig1_architecture")


# ---------------------------------------------------------------
# FIGURE 2 — Differentiation
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")
ax.set_facecolor(C_BG)

ax.text(50, 67, "Positioning of OmniGene-4-MM relative to recent MoE bio-models",
        ha="center", fontsize=12.5, weight="bold")

cards = [
    {
        "title": "AIDO.Protein\nSun et al. 2024",
        "x": 4,  "w": 28,
        "accent": C_GRAY2,
        "modal": "Protein only",
        "bullets": [
            "16B-A8B sparse MoE,\n  trained from scratch",
            "1.2 trillion tokens",
            "256 × A100, 64 days",
            "≈ 16,384 GPU-days",
            "Goal: SOTA on\n  PEER / ProteinGym",
        ],
        "footer": "≡ specialized tool",
    },
    {
        "title": "Tripathi et al. 2025\nSci Reports",
        "x": 36, "w": 28,
        "accent": C_GRAY2,
        "modal": "DNA only",
        "bullets": [
            "MoE-of-CNN ensemble\n  (DeepBIND backbone)",
            "Task: TFBS binary\n  classification",
            "ShiftSmooth XAI\n  attribution",
            "≈ 10 GPU-days",
            "Goal: stable TFBS\n  classifier",
        ],
        "footer": "≡ specialized tool",
    },
    {
        "title": "OmniGene-4-MM\n(this work)",
        "x": 68, "w": 28,
        "accent": C_ACCENT,
        "modal": "Vision · Sequence · Language\n8 modalities total",
        "bullets": [
            "Gemma-4 26B-A4B MoE\n  + LoRA r=64",
            "≈ 100 M tokens\n  (PAWS-X paraphrase)",
            "Single H20, 30 h\n  ≈ 1.25 GPU-days",
            "Modality-invariant\n  syntactic transfer",
            "Router-level XAI\n  (3-tier emergence)",
        ],
        "footer": "≡ mechanism + tool",
    },
]

for card in cards:
    x = card["x"]; w = card["w"]
    is_ours = card["accent"] == C_ACCENT
    body_fc = "#eef5fc" if is_ours else "white"
    ec = card["accent"]; lw = 1.6 if is_ours else 0.9

    p = FancyBboxPatch((x, 6), w, 56,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        fc=body_fc, ec=ec, lw=lw)
    ax.add_patch(p)

    p2 = FancyBboxPatch((x, 56), w, 6,
        boxstyle="round,pad=0.0,rounding_size=1.2",
        fc=ec, ec=ec, lw=lw)
    ax.add_patch(p2)
    ax.text(x+w/2, 59, card["title"], ha="center", va="center",
            fontsize=10.2, weight="bold", color="white")

    ax.text(x+w/2, 51, card["modal"], ha="center", va="center",
            fontsize=10.5, weight="bold",
            color=(C_ACCENT if is_ours else C_TEXT))

    for i, b in enumerate(card["bullets"]):
        ax.text(x+1.5, 45-i*7, "• " + b, ha="left", va="top",
                fontsize=8, color=C_TEXT)

    p3 = FancyBboxPatch((x+1, 7.5), w-2, 4,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        fc=("#dde8f5" if is_ours else "#f0efe8"),
        ec=ec, lw=0.7)
    ax.add_patch(p3)
    ax.text(x+w/2, 9.5, card["footer"], ha="center", va="center",
            fontsize=8.7,
            weight=("bold" if is_ours else "normal"),
            color=(C_ACCENT if is_ours else C_TEXT))

# Compute bar
ax.text(50, 3.8, "Training compute (log scale, GPU-days)",
        ha="center", fontsize=8.3, color=C_TEXT, style="italic")
ax.plot([18, 50, 82], [1.5, 1.5, 1.5], "k-", lw=0.4)
for x, val, color in [(18, 16384, C_GRAY2), (50, 10, C_GRAY2), (82, 1.25, C_ACCENT)]:
    ax.plot(x, 1.5, "o", color=color, markersize=8)
    ax.text(x, 0.2, f"{val:,} GPU-days" if val >= 1 else f"{val} GPU-days",
            ha="center", fontsize=7.7, weight="bold", color=color)

plt.savefig(f"{OUT}/fig2_differentiation.pdf")
plt.savefig(f"{OUT}/fig2_differentiation.png", dpi=200)
plt.close()
print("wrote fig2_differentiation")
print(f"all outputs in {OUT}")
