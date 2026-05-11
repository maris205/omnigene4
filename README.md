# OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability

> **TL;DR**  
> A Gemma-4-26B-A4B (128 experts, top-8 routing) fine-tuned into a unified DNA/protein/structure/natural-language bio-foundation model.  
> **98 % of cross-task expert differentiation comes from continued pretraining (CPT), only 2 % from supervised fine-tuning (SFT)** — a clean `CPT = representation / SFT = output alignment` factorization measured at the router level.

Paper: [`paper/omnigene4.pdf`](paper/omnigene4.pdf)

## Highlights

| Benchmark | Gemma-4-Instruct (baseline) | OmniGene-4 v2 | **OmniGene-4 v3** |
|---|---|---|---|
| BioPAWS Standard Homology (6k pairs) | 85 % | **100.00 %** | 99.95 % |
| BioPAWS Remote Homology (2k pairs) | 60 % | 56.55 % | **59.50 %** |
| BixBench Knowledge (True/False) | 87 % | 91.04 % | **93.66 %** |

Router-level measurements across 8 task pools × 3 checkpoints × 30 layers × 128 experts:

- **Baseline → CPT: ΔJS +0.092 (98 % of total)**
- **CPT → SFT+remote: ΔJS +0.002 (2 %)**
- Peak differentiation layer: **L12** (JS 0.082 → 0.251, 3×)
- SFT's gain is concentrated at **L28–L29** (near `lm_head`)

Five interpretable single-function atomic experts emerge after CPT:

| Expert | Function | Purity | Top tokens |
|---|---|---|---|
| E54 | English function-word | **80 %** NL | `the`, `.`, `,`, `in`, `to` |
| E59 | DNA dinucleotide | 46 % DNA | `AT`, `CT`, `AG`, `TT` |
| E117 | DNA dinucleotide (complementary) | 35 % DNA | `AG`, `AT`, `CT`, `TT` |
| E78 | Amino-acid single-letter | 36 % Protein | `V`, `K`, `G`, `Y` |
| E97 | Cellular-biology semantic | 19 % Cell | `cell`, `identity`, `type` |

## Repository structure

```
omnigene4/
├── paper/              # Final LaTeX paper + PDF + figures
├── training/           # CPT and SFT training scripts
├── evaluation/         # Benchmark evaluation scripts
├── analysis/           # MoE router-level analysis pipeline
├── results/            # Benchmark JSONs + MoE routing matrices
└── README.md
```

### `paper/`
- `omnigene4.pdf` — 18-page preprint, 51 references
- `omnigene4.tex` + `refs.bib` — LaTeX source
- `figures/` — all PDF figures

### `training/`
- `1-prepare_cpt_data_mp.py` — 22-process CPT corpus preprocessor (DNA + protein + 3Di + DSSP + text)
- `2-run_cpt.py` — 8-GPU DDP QLoRA CPT on Gemma-4-26B-A4B
- `12-merge_sft_v2.py` — 199 k-row instruction dataset assembly
- `14-train_bio_sft_v2.py` — single-GPU SFT from CPT checkpoint
- `16-build_remote_sft.py` — remote-homology augmentation (20 k pairs)
- `17-train_bio_sft_v3_remote.py` — SFT v3 from v2 checkpoint with added remote pairs

### `evaluation/`
- `15-eval_v2_sft.py` — evaluates v2 on Standard / Remote / BixBench
- `18-eval_v3_sft.py` — evaluates v3 on same protocol
- `15b-eval_v2_sft_remote.py` — standalone remote-only evaluation

### `analysis/`
- `20-collect_moe_activations.py` — forward-hook based router activation collector (`--tag {baseline, cpt, v3}`)
- `21-plot_moe_heatmaps.py` — task × expert heatmaps + delta + JS matrices
- `22-analyze_per_layer.py` — per-layer entropy + JS curves
- `23-three_way_compare.py` — baseline vs CPT vs v3 decomposition (the core analysis)
- `24-token_level_experts.py` — token-level purity and interpretability

### `results/`
- `benchmarks/omnigene4_v{2,3}_sft_eval.json` — final benchmark accuracies
- `moe_analysis/moe_counts_{baseline,cpt,v3}.npz` — raw per-layer per-expert routing counts (30 × 128 each)
- `moe_analysis/{report,per_layer_report,three_way_report}.json` — computed metrics
- `moe_analysis/*.png` — publication figures

## Methodology in one page

1. **Vocabulary injection**: 28 028 tokens added to Gemma-4-26B-A4B-Instruct (20 k DNA BPE + 8 k protein BPE + 20 Foldseek 3Di + 8 DSSP + control). Embeddings mean-initialized from BPE fragments. Vocab: 262 144 → 290 172.
2. **CPT**: 32.5 GB mixed corpus (DNA + protein + OpenWebText + 3Di/DSSP + instruction-replay), 0.6 epoch, 8× H20 with QLoRA (r=64, α=128, 8 target modules including `router.proj`). ~100 GPU-hours.
3. **SFT v2**: 179 k `{instruction, input, output}` triples across 8 categories (homology, UniProtQA, structure, mutation, cell, molecule, + replay). Single H20, 1 epoch. 11.8 h.
4. **SFT v3**: +20 k remote-homology pairs added to SFT. Lower LR to preserve v2 capabilities. 13.2 h.
5. **Router audit**: forward hooks on all 30 routers. For 8 task pools × 50 samples, record top-8 expert index per token. Compute per-layer cross-task JS divergence, per-task entropy, specialty scores, token-level purity.

## Reproduction

Not a turnkey release — the LoRA adapters and embedding checkpoints (~1.9 GB) are available on reasonable request. The scripts assume:

- **Hardware**: 1–8 × NVIDIA H20 (96 GB), CUDA 12.1+
- **Python 3.10+**, PyTorch 2.x
- **Dependencies**: `transformers>=4.35, peft, bitsandbytes, datasets, biopython, scikit-learn`
- **Blackwell fix**: scripts include the `transformers.integrations.moe._can_use_grouped_mm = lambda *a, **k: False` monkey-patch required for `sm_120` GPUs

Expected paths are currently hardcoded (e.g. `/root/autodl-tmp/dnagpt/...`). Adapt `BASE_MODEL`, `CPT_DIR`, `SFT_DIR` at the top of each script.

## Citation

```bibtex
@misc{wang2026omnigene4,
  author    = {Wang, Liang},
  title     = {{OmniGene-4}: A Unified Bio-Language {MoE} Model with Router-Level Interpretability},
  year      = {2026},
  howpublished = {\url{https://github.com/maris205/omnigene4}}
}
```

## License

Apache-2.0. See `LICENSE`.
