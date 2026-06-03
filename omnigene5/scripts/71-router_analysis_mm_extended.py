#!/usr/bin/env python
# coding: utf-8
"""
71-router_analysis_mm_extended.py

Extended router analysis on OmniGene-4-MM Stage 2 with 8 modalities:

Vision (4):
  1. vision_molecule  -- Vis-CheBI20 chemical structure images
  2. vision_medical   -- PubMedVision clinical images
  3. vision_pathology -- HPA10M tissue/pathology images
  4. vision_chart     -- ChartQA chart images

Sequence (3):
  5. protein_sequence -- amino acid sequences (homology pair prompts)
  6. dna_sequence     -- DNA bases (random ACGT prompts)
  7. structure_3di    -- 3Di structural alphabet (Foldseek)

Language (1):
  8. natural_language -- English biology questions

Output:
  /root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2/router_analysis_8mod/
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import random
import tarfile
import tempfile
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, inject_adapter_in_model

BASE_MODEL = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR     = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage2"
DATA_DIR   = "/root/autodl-tmp/dnagpt/omnigene5/data"
EVAL_FILE  = "/root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl"
OUT_DIR    = Path(MM_DIR) / "router_analysis_8mod"
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

N_PROMPTS = 50
N_LAYERS = 30
N_EXPERTS = 128

random.seed(42)

print("=" * 60)
print("OmniGene-4-MM extended router analysis (8 modalities)")
print("=" * 60)


# ============== Load model ==============
print("\n[1/5] Loading model...", flush=True)
processor = AutoProcessor.from_pretrained(MM_DIR)
tokenizer = AutoTokenizer.from_pretrained(MM_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
)
lora_cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj','router.proj'])
inject_adapter_in_model(lora_cfg, model.model.language_model, adapter_name="stage2")
ms = model.state_dict()
for k, v in torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu").items():
    if k in ms: ms[k].copy_(v)
model.get_input_embeddings().weight.data.copy_(
    torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu"))
model.eval()
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ============== Hooks ==============
expert_counts = {}
current = {"task": None}


def make_hook(layer_idx):
    def hook(module, inputs, output):
        if not isinstance(output, tuple) or len(output) < 3: return
        idx = output[2].detach().cpu().numpy()
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


print("\n[2/5] Installing hooks...", flush=True)
hooks = []
for i, layer in enumerate(model.model.language_model.layers):
    if hasattr(layer, "router"):
        hooks.append(layer.router.register_forward_hook(make_hook(i)))
print(f"  attached on {len(hooks)}/30 routers")


# ============== Build prompt pools ==============
print("\n[3/5] Building prompt pools...", flush=True)

prompts = {}

# A. vision_molecule (Vis-CheBI20)
chebi_base = f"{DATA_DIR}/B_chebi20"
with open(f"{chebi_base}/test.json") as f:
    chebi = json.load(f)
mol_picks = random.sample(chebi, N_PROMPTS * 2)
mol_p = []
for r in mol_picks:
    img_path = f"{chebi_base}/{r['images'][0]}"
    if os.path.exists(img_path) and len(mol_p) < N_PROMPTS:
        user = r["messages"][0]["content"].replace("<image>", "").strip()
        mol_p.append((img_path, user))
prompts["vision_molecule"] = mol_p
print(f"  vision_molecule: {len(mol_p)}")

# B. vision_medical (PubMedVision)
import ijson
med_pool = []
with open(f"{DATA_DIR}/J_biomedvis/synthetic_visual_tasks.json", "rb") as f:
    for item in ijson.items(f, "item"):
        if len(med_pool) >= 200: break
        rel = item.get("images", [None])[0]
        if not rel: continue
        img_path = f"{DATA_DIR}/D_pubmedvision/images_extracted/images/{rel}"
        if not os.path.exists(img_path): continue
        msg = item.get("messages", [])
        if not msg: continue
        user_txt = msg[0]["content"].replace("<image>","").strip()
        med_pool.append((img_path, user_txt))
prompts["vision_medical"] = random.sample(med_pool, min(N_PROMPTS, len(med_pool)))
print(f"  vision_medical: {len(prompts['vision_medical'])}")

# C. vision_pathology (HPA10M)
hpa_dir = f"{DATA_DIR}/C_hpa_microscopy/hpa10m_train"
hpa_pool = []
hpa_extract_dir = Path(OUT_DIR) / "_hpa_imgs"
hpa_extract_dir.mkdir(exist_ok=True)
shards = sorted(os.listdir(hpa_dir))[:2]  # only need 2 shards for 50 samples
for shard in shards:
    if len(hpa_pool) >= N_PROMPTS * 2: break
    try:
        t = tarfile.open(f"{hpa_dir}/{shard}")
        members = {}
        for m in t.getmembers():
            stem = m.name.rsplit(".", 1)[0]
            ext = m.name.rsplit(".", 1)[-1]
            members.setdefault(stem, {})[ext] = m
        for stem, parts in members.items():
            if len(hpa_pool) >= N_PROMPTS * 2: break
            if "jpg" not in parts or "json" not in parts: continue
            jf = t.extractfile(parts["json"])
            if not jf: continue
            meta = json.loads(jf.read())
            jpg_data = t.extractfile(parts["jpg"])
            if not jpg_data: continue
            img_path = hpa_extract_dir / f"{stem.replace('/', '_')}.jpg"
            if not img_path.exists():
                with open(img_path, "wb") as fp: fp.write(jpg_data.read())
            gene = meta.get("gene_name", "this protein")
            hpa_pool.append((str(img_path),
                             f"What protein is shown in this Human Protein Atlas image and where is it localized?"))
        t.close()
    except Exception as e:
        print(f"  HPA shard {shard} err: {e}")
prompts["vision_pathology"] = random.sample(hpa_pool, min(N_PROMPTS, len(hpa_pool)))
print(f"  vision_pathology: {len(prompts['vision_pathology'])}")

# D. vision_chart (ChartQA)
import pyarrow.parquet as pq
chart_dir = f"{DATA_DIR}/E_chartqa/data"
chart_pool = []
chart_extract = Path(OUT_DIR) / "_chart_imgs"
chart_extract.mkdir(exist_ok=True)
for pf in sorted(Path(chart_dir).glob("train-*.parquet"))[:1]:
    f = pq.ParquetFile(pf)
    for batch in f.iter_batches(batch_size=128):
        if len(chart_pool) >= N_PROMPTS * 2: break
        for row in batch.to_pylist():
            if len(chart_pool) >= N_PROMPTS * 2: break
            img_b = row.get("image", {}).get("bytes")
            if not img_b: continue
            img_path = chart_extract / f"chart_{len(chart_pool):04d}.png"
            if not img_path.exists():
                with open(img_path, "wb") as fp: fp.write(img_b)
            q = row.get("query", "What does this chart show?")
            chart_pool.append((str(img_path), q))
prompts["vision_chart"] = random.sample(chart_pool, min(N_PROMPTS, len(chart_pool)))
print(f"  vision_chart: {len(prompts['vision_chart'])}")

# E. protein_sequence (homology pair-style)
from datasets import load_dataset
ds_prot = load_dataset("dnagpt/biopaws", "protein_pair_short", split="train")
prot_picks = random.sample(list(ds_prot), N_PROMPTS)
prompts["protein_sequence"] = [
    f"### Instruction:\nDetermine if the two sequences below are structurally related (like paraphrases).\n\n### Sequence 1:\n{p['sentence1']}\n\n### Sequence 2:\n{p['sentence2']}\n\n### Answer:\n"
    for p in prot_picks
]
print(f"  protein_sequence: {len(prompts['protein_sequence'])}")

# F. dna_sequence (random ACGT)
dna_p = []
for _ in range(N_PROMPTS):
    seq = "".join(random.choice("ACGT") for _ in range(120))
    dna_p.append(f"### Instruction:\nDescribe the DNA sequence motifs.\n\n### Input:\n{seq}\n\n### Answer:\n")
prompts["dna_sequence"] = dna_p
print(f"  dna_sequence: {len(dna_p)}")

# G. structure_3di (from Structure category in eval set)
struct_pool = []
with open(EVAL_FILE) as f:
    for line in f:
        r = json.loads(line)
        if r.get("category") == "Structure":
            struct_pool.append(r)
struct_picks = random.sample(struct_pool, min(N_PROMPTS, len(struct_pool)))
prompts["structure_3di"] = []
for r in struct_picks:
    instr = r["instruction"]
    inp = r.get("input", "")
    if inp.strip():
        prompt = f"### Instruction:\n{instr}\n\n{inp}\n\n### Answer:\n"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Answer:\n"
    prompts["structure_3di"].append(prompt)
print(f"  structure_3di: {len(prompts['structure_3di'])}")

# H. natural_language
nl_questions = [
    "What is the function of the hemoglobin protein?",
    "Describe the role of mitochondria in eukaryotic cells.",
    "Explain how DNA replication works.",
    "What is a transcription factor and how does it work?",
    "Describe the central dogma of molecular biology.",
    "What is the difference between mitosis and meiosis?",
    "How do enzymes catalyze biochemical reactions?",
    "What is the structure of an antibody?",
    "Explain the citric acid cycle in cell respiration.",
    "Describe the role of ribosomes in protein synthesis.",
] * 5
random.shuffle(nl_questions)
prompts["natural_language"] = [
    f"### Instruction:\n{q}\n\n### Answer:\n" for q in nl_questions[:N_PROMPTS]
]
print(f"  natural_language: {len(prompts['natural_language'])}")


# ============== Forward pass ==============
print("\n[4/5] Forward passes...", flush=True)


def run_image(image_path, user_text, task_name):
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": user_text}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = processor(text=text, images=[img], return_tensors="pt").to(model.device)
    current["task"] = task_name
    with torch.no_grad():
        _ = model(**inp, use_cache=False)


def run_text(prompt, task_name):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    current["task"] = task_name
    with torch.no_grad():
        _ = model(input_ids=ids, use_cache=False)


for task, items in prompts.items():
    print(f"\n  >> {task} ({len(items)} prompts)")
    for i, item in enumerate(items):
        if i % 10 == 0: print(f"     {i}/{len(items)}", flush=True)
        try:
            if task.startswith("vision_"):
                run_image(item[0], item[1], task)
            else:
                run_text(item, task)
        except Exception as e:
            print(f"     err: {type(e).__name__}: {str(e)[:80]}")

for h in hooks: h.remove()
print(f"\n  collected: {sorted(expert_counts.keys())}")


# ============== Compute JS + save ==============
print("\n[5/5] Computing JS + saving...", flush=True)


def js(p, q):
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


for task, mat in expert_counts.items():
    np.savez_compressed(OUT_DIR / f"moe_counts_{task}.npz", counts=mat)

modalities = sorted(expert_counts.keys())
report = {
    "modalities": modalities,
    "n_layers": N_LAYERS,
    "n_experts": N_EXPERTS,
    "n_prompts_per_modality": N_PROMPTS,
    "layer_averaged_pairwise_js": {},
    "expert_specialty_top5": {},
}

for i, t1 in enumerate(modalities):
    for j, t2 in enumerate(modalities):
        if i >= j: continue
        avg = np.mean([js(expert_counts[t1][L], expert_counts[t2][L]) for L in range(N_LAYERS)])
        report["layer_averaged_pairwise_js"][f"{t1}_vs_{t2}"] = float(avg)

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


# ============ Plots ============
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 8x8 layer-averaged JS heatmap
fig, ax = plt.subplots(figsize=(9, 8))
mat = np.zeros((len(modalities), len(modalities)))
for i, t1 in enumerate(modalities):
    for j, t2 in enumerate(modalities):
        if i == j: mat[i,j] = 0.0
        elif i < j:
            avg = np.mean([js(expert_counts[t1][L], expert_counts[t2][L]) for L in range(N_LAYERS)])
            mat[i,j] = avg
            mat[j,i] = avg
im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=mat.max())
ax.set_xticks(range(len(modalities)))
ax.set_yticks(range(len(modalities)))
ax.set_xticklabels(modalities, rotation=45, ha="right")
ax.set_yticklabels(modalities)
for i in range(len(modalities)):
    for j in range(len(modalities)):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if mat[i,j] > mat.max() * 0.5 else "black")
ax.set_title("OmniGene-4-MM: pairwise routing JS divergence (8 modalities, layer-averaged)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "js_heatmap_8mod.pdf", dpi=150)
plt.savefig(OUT_DIR / "figures" / "js_heatmap_8mod.png", dpi=150)
plt.close()

# Modality x expert (8 modalities)
fig, ax = plt.subplots(figsize=(14, 6))
mat_e = np.stack([expert_counts[t][L_focus] for t in modalities], axis=0)
mat_e = mat_e / (mat_e.sum(axis=1, keepdims=True) + 1e-10)
im = ax.imshow(mat_e, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(modalities)))
ax.set_yticklabels(modalities)
ax.set_xlabel("Expert ID (0-127)")
ax.set_title(f"Modality x Expert routing at Layer {L_focus} (normalized)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / f"heatmap_8mod_L{L_focus}.pdf", dpi=150)
plt.savefig(OUT_DIR / "figures" / f"heatmap_8mod_L{L_focus}.png", dpi=150)
plt.close()

# Per-layer JS curves (just key pairs)
fig, ax = plt.subplots(figsize=(11, 6))
key_pairs = [
    ("vision_molecule", "protein_sequence"),
    ("vision_pathology", "natural_language"),
    ("dna_sequence", "protein_sequence"),
    ("vision_medical", "vision_pathology"),
    ("structure_3di", "protein_sequence"),
]
for t1, t2 in key_pairs:
    if t1 not in expert_counts or t2 not in expert_counts: continue
    ys = [js(expert_counts[t1][L], expert_counts[t2][L]) for L in range(N_LAYERS)]
    ax.plot(range(N_LAYERS), ys, label=f"{t1} vs {t2}", marker='o', markersize=3)
ax.set_xlabel("Layer")
ax.set_ylabel("Routing JS divergence")
ax.set_title("OmniGene-4-MM: per-layer routing JS (key modality pairs)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "per_layer_js_8mod.pdf", dpi=150)
plt.savefig(OUT_DIR / "figures" / "per_layer_js_8mod.png", dpi=150)
plt.close()

print(f"\nReport saved to {OUT_DIR}/routing_report.json")
print(f"Figures saved to {OUT_DIR}/figures/")
print()
print("=== Layer-averaged pairwise JS (sorted) ===")
for k, v in sorted(report["layer_averaged_pairwise_js"].items(), key=lambda x: -x[1]):
    print(f"  {k:65s}: {v:.4f}")
