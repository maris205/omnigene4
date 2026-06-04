# OmniGene-4-MM: multi-modal extension of OmniGene-4

This directory contains the training, evaluation, analysis, and figure-generation
scripts for the multi-modal extension reported in the manuscript:

> **OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability
> and Modality-Invariant Transfer**

The pre-MM (v5) pipeline lives in `../biopaws/`.

## Quick map

| File | What it does | Where it appears in the paper |
|---|---|---|
| `scripts/01-build_unified_jsonl.py` | Build the unified multi-modal training corpus (vision + sequence + text). | §Methods (data preparation) |
| `scripts/02-fix_image_paths.py` | Resolve relative image paths after corpus assembly. | §Methods |
| `scripts/20-vision_lora_smoke_test.py` | Smoke test that the vision tower forward-passes correctly. | (sanity check, not in paper) |
| `scripts/30-train_omnigene5_stage1.py` | **Stage 1** — vision-only LoRA warmup. | §Methods (Stage 1) |
| `scripts/40-eval_omnigene4mm.py` | Generic MM evaluator. | (helper) |
| `scripts/50-train_stage2.py` | **Stage 2** — mixed text + vision SFT, recovers catastrophic forgetting. | §Methods (Stage 2) |
| `scripts/60-eval_stage2.py` | Stage 2 evaluation (vision + homology + multi-task gen). | §Results, Table 2 |
| `scripts/70-router_analysis_mm.py` | 5-modality router-level analysis (initial). | (predecessor of 71) |
| `scripts/71-router_analysis_mm_extended.py` | **8-modality router analysis** with JS divergence. | §Results, modality-invariant transfer |
| `scripts/80-train_stage3.py` | Stage 3 v1 (failed — adapter naming bug, kept for reproducibility). | §Methods Footnote |
| `scripts/82-train_stage3v2.py` | **Stage 3 v2** — first repaired attempt, plateaued at homology 59%. | §Methods (failure mode) |
| `scripts/83-train_stage3v3.py` | **Stage 3 v3 (final)** — LR 2e-5, frozen embedding, 3000 steps. | §Methods (final stage) |
| `scripts/90-eval_stage3.py` | Stage 3 v1 evaluation (legacy). | (legacy) |
| `scripts/91-eval_stage3v2.py` | Stage 3 v2 evaluation. | §Results, Table 2 |
| `scripts/92-eval_stage3v3.py` | **Stage 3 v3 evaluation (final)**. | §Results, Tables 1 + 2 |
| `scripts/95-qualitative_demo_stage3v3.py` | Qualitative showcase: vision + homology + multi-task gen. | §Results, Figure 4 |
| `scripts/96-upload_to_hf.py`              | One-shot uploader for v5-merged + MM-LoRA HF repos. | (release tooling) |
| `scripts/97-upload_mm_robust.py`          | Per-file retry uploader (used because hf-xet died on long uploads). | (release tooling) |
| `scripts/98-merge_mm_v3.py`               | **Merge MM v3 LoRA + embedding INTO v5-merged** to produce a stand-alone BF16 multi-modal checkpoint. Requires GPU. | §Methods (model release) |
| `scripts/99-upload_merged.sh`             | Bash uploader for the 49 GB merged checkpoint (11 safetensors shards). | (release tooling) |
| `scripts/99b-retry_missing_shards.sh`     | Aggressive retry on shards that silently fail mid-upload. | (release tooling) |

## Figures

`figures_paper_mm/` contains the four MM figures from the merged manuscript
plus their generation scripts:

| File | Description |
|---|---|
| `fig1_architecture.pdf` | OmniGene-4-MM architecture (vision tower + LoRA-injected MoE backbone + 3 heads + 3-stage pipeline) |
| `fig2_differentiation.pdf` | Positioning vs AIDO.Protein (Sun et al. 2024) and Tripathi et al. (2025) |
| `fig3_stage_progression.pdf` | Capability progression across all training stages (homology, vision, multi-task gen) |
| `fig4_qualitative.pdf` | 8-panel qualitative output across modalities |
| `make_figures.py` | Generates fig1 + fig2 |
| `make_figures_b.py` | Generates fig3 + fig4 |
| `tables.tex` | LaTeX source for Tables 1, 2, 3 |
| `manuscript_outline.md` | Section-level outline of the merged paper |

## Reproducing the MM pipeline

The full pipeline costs about 1.5 GPU-days on a single H20.

```bash
# Prerequisites: v5 merged checkpoint at /root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged
# (or download from https://huggingface.co/dnagpt/OmniGene-4-SFT-v5-merged)

# 1. Build the unified MM training corpus
python scripts/01-build_unified_jsonl.py
python scripts/02-fix_image_paths.py

# 2. Stage 1: vision warmup (~10K steps, ~0.4 GPU-days)
python scripts/30-train_omnigene5_stage1.py

# 3. Stage 2: mixed recovery (~6K steps, ~1 GPU-day)
python scripts/50-train_stage2.py

# 4. Stage 3 v3: homology specialty (~3K steps, ~0.5 GPU-days)
python scripts/83-train_stage3v3.py

# 5. Evaluation
python scripts/92-eval_stage3v3.py    # full benchmark
python scripts/95-qualitative_demo_stage3v3.py  # Figure 4 source

# 6. Router analysis (Figure 5 source)
python scripts/71-router_analysis_mm_extended.py
```

## Outputs

Each training script saves:
- `lora_weights.pt` — LoRA adapter state-dict (≈160 MB)
- `embedding_weights.pt` — extended embedding table (≈1.6 GB)
- `meta.json` — training hyperparameters

Evaluation scripts save `eval_report.json` in the same directory as the model.

The final OmniGene-4-MM Stage 3 v3 weights are released at
<https://huggingface.co/dnagpt/OmniGene-4-MM-LoRA>.

## Datasets

| Modality | Source |
|---|---|
| Chemical structure | [Vis-CheBI20](https://huggingface.co/datasets/PharMolix/Vis-CheBI20) |
| Medical image | [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) |
| Pathology | HPA10M (Human Protein Atlas) |
| Charts | [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) |
| Synthetic biomed visual | (project-internal) |
| Protein homology | [BioPAWS](https://huggingface.co/datasets/dnagpt/biopaws) |
| Multi-task SFT | (project-internal, `omnigene_sft_v1_*.jsonl`) |

## Citation

```bibtex
@article{wang2026omnigene4mm,
  title  = {OmniGene-4: A Unified Bio-Language MoE Model with Router-Level
            Interpretability and Modality-Invariant Transfer},
  author = {Wang, Liang},
  year   = {2026},
  note   = {Manuscript under review at Patterns (Cell Press).
            Preliminary version: bioRxiv 10.1101/2026.01.03.697478}
}
```

## License

Code: MIT. Model weights: see Hugging Face model cards. Datasets: respective
licenses on Hugging Face.
