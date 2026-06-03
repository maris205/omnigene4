#!/usr/bin/env python
# coding: utf-8
"""
70-router_analysis_mm.py

Router-level analysis on OmniGene-4-MM Stage 2.

Collect routing across 5 modality categories:
1. vision_molecule  -- Vis-CheBI20 molecule images (50 prompts)
2. vision_medical   -- PubMedVision medical images (50 prompts)
3. protein_sequence -- protein homology pair prompts (50)
4. dna_sequence     -- DNA pair prompts (50)
5. natural_language -- BixBench-style English questions (50)

For each prompt, install forward hooks on all 30 routers and record top-8 expert
activations. Aggregate to per-task expert distribution (50 prompts averaged),
then compute pairwise JS divergence between modality categories at each layer.

Output:
  /root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2/router_analysis/
    moe_counts_<modality>.npz   -- 30x128 routing count matrix
    routing_report.json         -- per-layer JS, decomposition stats
    figures/                    -- heatmap + per-layer JS curves
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, inject_adapter_in_model

BASE_MODEL = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR     = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2"
DATA_DIR   = "/root/autodl-tmp/dnagpt/omnigene5/data"
OUT_DIR    = Path(MM_DIR) / "router_analysis"
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

N_PROMPTS_PER_MODALITY = 50
N_LAYERS = 30
N_EXPERTS = 128
TOP_K = 8

random.seed(42)

print("=" * 60)
print("OmniGene-4-MM Stage 2 router-level analysis")
print("=" * 60)


# ============== Load model ==============
print("\n[1/5] Loading model...", flush=True)
processor = AutoProcessor.from_pretrained(MM_DIR)
tokenizer = AutoTokenizer.from_pretrained(MM_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
lora_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj','router.proj'],
)
inject_adapter_in_model(lora_cfg, model.model.language_model, adapter_name="stage2")
ms = model.state_dict()
mm_lora = torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu")
loaded = 0
for k, v in mm_lora.items():
    if k in ms: ms[k].copy_(v); loaded += 1
print(f"  loaded {loaded} LoRA tensors")
mm_emb = torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu")
model.get_input_embeddings().weight.data.copy_(mm_emb)
model.eval()
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ============== Build hooks on all 30 routers ==============
print("\n[2/5] Installing hooks...", flush=True)
expert_counts = {}  # task_name -> [N_LAYERS, N_EXPERTS]
current = {"task": None}


def make_hook(layer_idx):
    def hook(module, inputs, output):
        if not isinstance(output, tuple) or len(output) < 3: return
        # Gemma4Router output: (router_logits, top_k_weights, top_k_indices)
        idx = output[2].detach().cpu().numpy()  # [n_tokens, top_k]
        task = current["task"]
        if task is None: return
        if task not in expert_counts:
            expert_counts[task] = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.float32)
        for tok in range(idx.shape[0]):
            for k in range(idx.shape[1]):
                e = int(idx[tok, k])
                if 0 <= e < N_EXPERTS:
                    expert_counts[task][layer_idx, e] += 1
    return hook


hooks = []
language_model = model.model.language_model
n_attached = 0
for i, layer in enumerate(language_model.layers):
    if hasattr(layer, "router"):
        h = layer.router.register_forward_hook(make_hook(i))
        hooks.append(h)
        n_attached += 1
print(f"  attached on {n_attached}/{len(language_model.layers)} routers")


# ============== Collect prompts for 5 modalities ==============
print("\n[3/5] Building prompt pools...", flush=True)

# A. vision_molecule (Vis-CheBI20)
chebi_path = f"{DATA_DIR}/B_chebi20"
with open(f"{chebi_path}/test.json") as f:
    chebi = json.load(f)
mol_samples = random.sample(chebi, N_PROMPTS_PER_MODALITY)
mol_prompts = []
for r in mol_samples:
    img_path = f"{chebi_path}/{r['images'][0]}"
    if not os.path.exists(img_path): continue
    user = r["messages"][0]["content"].replace("<image>", "").strip()
    mol_prompts.append((img_path, user))
print(f"  vision_molecule: {len(mol_prompts)}")

# B. vision_medical (PubMedVision -- pick from synthetic_visual_tasks subset)
import ijson
medical_pool = []
with open(f"{DATA_DIR}/J_biomedvis/synthetic_visual_tasks.json", "rb") as f:
    for item in ijson.items(f, "item"):
        if len(medical_pool) >= 200: break
        rel = item.get("images", [None])[0]
        if not rel: continue
        # Resolve via path-fix (image lives in PubMedVision images_extracted/images/)
        img_path = f"{DATA_DIR}/D_pubmedvision/images_extracted/images/{rel}"
        if not os.path.exists(img_path):
            img_path = f"{DATA_DIR}/D_pubmedvision/images_extracted/{rel}"
            if not os.path.exists(img_path): continue
        msg = item.get("messages", [])
        if not msg: continue
        user_txt = msg[0]["content"].replace("<image>","").strip()
        medical_pool.append((img_path, user_txt))
medical_prompts = random.sample(medical_pool, min(N_PROMPTS_PER_MODALITY, len(medical_pool)))
print(f"  vision_medical: {len(medical_prompts)}")

# C. protein_sequence (homology pair-style prompts)
from datasets import load_dataset
ds_prot = load_dataset("dnagpt/biopaws", "protein_pair_short", split="train")
prot_picks = random.sample(list(ds_prot), N_PROMPTS_PER_MODALITY)
prot_prompts = []
for p in prot_picks:
    txt = (
        "### Instruction:\nDetermine if the two sequences below are "
        "structurally related (like paraphrases).\n\n"
        f"### Sequence 1:\n{p['sentence1']}\n\n"
        f"### Sequence 2:\n{p['sentence2']}\n\n### Answer:\n"
    )
    prot_prompts.append(txt)
print(f"  protein_sequence: {len(prot_prompts)}")

# D. dna_sequence (use DNA-only prompts from existing OmniGene-4 SFT corpus, fallback random DNA)
dna_prompts = []
import random as _r
_alphabet = "ACGT"
for _ in range(N_PROMPTS_PER_MODALITY):
    seq = "".join(_r.choice(_alphabet) for _ in range(120))
    dna_prompts.append(f"### Instruction:\nDescribe the DNA sequence motifs.\n\n### Input:\n{seq}\n\n### Answer:\n")
print(f"  dna_sequence: {len(dna_prompts)}")

# E. natural_language (English biology questions)
nl_prompts = [
    "### Instruction:\nWhat is the function of the hemoglobin protein?\n\n### Answer:\n",
    "### Instruction:\nDescribe the role of mitochondria in eukaryotic cells.\n\n### Answer:\n",
    "### Instruction:\nExplain how DNA replication works.\n\n### Answer:\n",
    "### Instruction:\nWhat is a transcription factor and how does it work?\n\n### Answer:\n",
    "### Instruction:\nDescribe the central dogma of molecular biology.\n\n### Answer:\n",
] * 10
random.shuffle(nl_prompts)
nl_prompts = nl_prompts[:N_PROMPTS_PER_MODALITY]
print(f"  natural_language: {len(nl_prompts)}")


# ============== Forward pass and collect routes ==============
print("\n[4/5] Forward pass on all prompts (collecting routes)...", flush=True)


def run_image_prompt(image_path, user_text, task_name):
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": user_text},
    ]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = processor(text=text, images=[img], return_tensors="pt").to(model.device)
    current["task"] = task_name
    with torch.no_grad():
        _ = model(**inp, use_cache=False)


def run_text_prompt(prompt, task_name):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    current["task"] = task_name
    with torch.no_grad():
        _ = model(input_ids=ids, use_cache=False)


for i, (path, user) in enumerate(mol_prompts):
    if i % 10 == 0: print(f"  vision_molecule {i}/{len(mol_prompts)}", flush=True)
    try: run_image_prompt(path, user, "vision_molecule")
    except Exception as e:
        print(f"    err: {type(e).__name__}: {str(e)[:80]}")

for i, (path, user) in enumerate(medical_prompts):
    if i % 10 == 0: print(f"  vision_medical {i}/{len(medical_prompts)}", flush=True)
    try: run_image_prompt(path, user, "vision_medical")
    except Exception as e:
        print(f"    err: {type(e).__name__}: {str(e)[:80]}")

for i, p in enumerate(prot_prompts):
    if i % 10 == 0: print(f"  protein_sequence {i}/{len(prot_prompts)}", flush=True)
    try: run_text_prompt(p, "protein_sequence")
    except Exception as e:
        print(f"    err: {type(e).__name__}: {str(e)[:80]}")

for i, p in enumerate(dna_prompts):
    if i % 10 == 0: print(f"  dna_sequence {i}/{len(dna_prompts)}", flush=True)
    try: run_text_prompt(p, "dna_sequence")
    except Exception as e:
        print(f"    err: {type(e).__name__}: {str(e)[:80]}")

for i, p in enumerate(nl_prompts):
    if i % 10 == 0: print(f"  natural_language {i}/{len(nl_prompts)}", flush=True)
    try: run_text_prompt(p, "natural_language")
    except Exception as e:
        print(f"    err: {type(e).__name__}: {str(e)[:80]}")

# Remove hooks
for h in hooks: h.remove()
print(f"  collected {len(expert_counts)} modalities")


# ============== Compute JS divergence and save ==============
print("\n[5/5] Computing pairwise JS + saving...", flush=True)


def js(p, q):
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# Save raw counts
for task, mat in expert_counts.items():
    np.savez_compressed(OUT_DIR / f"moe_counts_{task}.npz", counts=mat)

# Per-layer pairwise JS between all modality pairs
modalities = sorted(expert_counts.keys())
report = {
    "modalities": modalities,
    "n_layers": N_LAYERS,
    "n_experts": N_EXPERTS,
    "n_prompts_per_modality": N_PROMPTS_PER_MODALITY,
    "per_layer_pairwise_js": {},
    "layer_averaged_pairwise_js": {},
    "expert_specialty_top5": {},
}

# Per-layer per-pair JS
for L in range(N_LAYERS):
    for i, t1 in enumerate(modalities):
        for j, t2 in enumerate(modalities):
            if i >= j: continue
            j_val = js(expert_counts[t1][L], expert_counts[t2][L])
            key = f"L{L:02d}_{t1}_vs_{t2}"
            report["per_layer_pairwise_js"][key] = float(j_val)

# Layer-averaged pairwise JS
for i, t1 in enumerate(modalities):
    for j, t2 in enumerate(modalities):
        if i >= j: continue
        avg = np.mean([js(expert_counts[t1][L], expert_counts[t2][L]) for L in range(N_LAYERS)])
        report["layer_averaged_pairwise_js"][f"{t1}_vs_{t2}"] = float(avg)

# Per-task top-5 specialist experts at L12 (peak modality differentiation layer)
L_focus = 12
for task in modalities:
    layer = expert_counts[task][L_focus]
    total = layer.sum()
    if total == 0: continue
    top_idx = np.argsort(-layer)[:5]
    report["expert_specialty_top5"][task] = [
        {"expert": int(e), "fraction": float(layer[e] / total)}
        for e in top_idx
    ]

with open(OUT_DIR / "routing_report.json", "w") as f:
    json.dump(report, f, indent=2)

# ============ Plot heatmaps + JS curves ============
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Per-layer pairwise JS curves
fig, ax = plt.subplots(figsize=(11, 6))
for i, t1 in enumerate(modalities):
    for j, t2 in enumerate(modalities):
        if i >= j: continue
        ys = [js(expert_counts[t1][L], expert_counts[t2][L]) for L in range(N_LAYERS)]
        ax.plot(range(N_LAYERS), ys, label=f"{t1} vs {t2}", marker='o', markersize=3)
ax.set_xlabel("Layer")
ax.set_ylabel("Routing JS divergence")
ax.set_title("OmniGene-4-MM: per-layer routing JS divergence between modality pairs")
ax.legend(loc="upper right", fontsize=7, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "per_layer_js.pdf", dpi=150)
plt.savefig(OUT_DIR / "figures" / "per_layer_js.png", dpi=150)
plt.close()

# 2. Modality x expert heatmap at L12
fig, ax = plt.subplots(figsize=(14, 5))
mat = np.stack([expert_counts[t][L_focus] for t in modalities], axis=0)
mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-10)
im = ax.imshow(mat, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(modalities)))
ax.set_yticklabels(modalities)
ax.set_xlabel("Expert ID (0-127)")
ax.set_title(f"Modality x Expert routing at Layer {L_focus} (normalized)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / f"heatmap_L{L_focus}.pdf", dpi=150)
plt.savefig(OUT_DIR / "figures" / f"heatmap_L{L_focus}.png", dpi=150)
plt.close()

# 3. Layer-averaged JS bar chart
fig, ax = plt.subplots(figsize=(11, 5))
keys = list(report["layer_averaged_pairwise_js"].keys())
vals = [report["layer_averaged_pairwise_js"][k] for k in keys]
order = np.argsort(-np.array(vals))
keys = [keys[o] for o in order]
vals = [vals[o] for o in order]
ax.barh(range(len(keys)), vals, color="#4E79A7")
ax.set_yticks(range(len(keys)))
ax.set_yticklabels(keys, fontsize=8)
ax.set_xlabel("Layer-averaged routing JS")
ax.set_title("OmniGene-4-MM: cross-modality routing differentiation")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "layer_avg_js_bars.pdf", dpi=150)
plt.savefig(OUT_DIR / "figures" / "layer_avg_js_bars.png", dpi=150)
plt.close()

print(f"\nReport saved to {OUT_DIR}/routing_report.json")
print(f"Figures saved to {OUT_DIR}/figures/")
print()
print("=== Summary: layer-averaged pairwise JS ===")
for k, v in sorted(report["layer_averaged_pairwise_js"].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")

print()
print("=== Top-5 experts per modality at L12 ===")
for t, top in report["expert_specialty_top5"].items():
    print(f"  {t}:")
    for x in top:
        print(f"    E{x['expert']:3d}: {x['fraction']*100:.1f}%")
