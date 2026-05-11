# Codex Pre-submission Review (Round 1)

**Date**: 2026-05-11
**Reviewer**: GPT-5.4 via Codex MCP at `xhigh` reasoning effort
**Thread ID**: `019e15ed-3797-7450-858e-455638d91739`
**Verdict**: 3/10, confidence 4/5. **Recommend major rework before bioRxiv post.**

---

## Top 5 rejection risks

1. **Layer-averaged expert IDs may be invalid.** Gemma-4-MoE has 128 *per-layer* experts. "E54 at L0" and "E54 at L12" are different parameters. Sections 3.3.4 and the atomic-experts table are based on layer-averaged specialty scores — this conflates non-comparable objects. **Verified in code (`21-plot_moe_heatmaps.py:54` does `mat.mean(axis=0)`).**
2. **Evaluation rests on the author's own benchmark** with no external validation, no leakage analysis between SFT data and BioPAWS eval splits, and no co-author/external-lab confirmation.
3. **Baseline definition mismatch**: routing-analysis "baseline" = Gemma + new vocab + mean-init embeddings (no LoRA), but benchmark table labels it "Gemma-4-26B-Instruct, zero-shot". These are different objects.
4. **Causal claims exceed evidence**: "98% CPT / 2% SFT" is one architecture, one seed, no prompt-bootstrap, no router-LoRA ablation data shown.
5. **Hard benchmark is weak**: v3 Remote 59.50% vs claimed Gemma baseline 60% = **−0.5**. Abstract says "outperforms by 0 points", which is false as written.

## Validity of JS divergence claims

- N=50 prompts is **insufficient** for the precision claimed. Effective sample = prompts, not tokens (within-prompt correlation).
- Baseline only valid for narrow question; not for "what biology CPT adds beyond Gemma".
- **Prompt-format confounds are major**: tasks have different templates (raw seq vs `### Sequence 1:` vs OpenWebText). Author's own E54/E100 token tables admit "DNA-like 2-grams leaking from homology prompts".

## Is "98% vs 2%" defensible?

Arithmetically yes (`(0.230-0.138)/0.094 = 0.98`). Scientifically no:
- No uncertainty intervals.
- Average JS suppresses targeted late-layer SFT effects.
- v3 bundles SFT+remote so stage attribution is coarse.
- Safe wording: "Under our average-JS metric, most of the observed routing-separation increase occurred during CPT".

## Minimum controls

1. **Ablation matrix**: untouched Gemma / vocab-only / CPT-no-router-LoRA / full CPT — separate tokenizer-expansion from CPT proper.
2. **Format-matched routing control**: strip `### Sequence` headers, swap templates across tasks. If JS survives, claim survives.
3. **Statistical robustness**: ≥3 seeds, prompt-bootstrap CIs, McNemar/paired bootstrap on v2 vs v3.
4. **External benchmarks** with same-split head-to-head: ESM-2, ProtT5/CATHe, TM-Vec/PLMSearch, MMseqs2/DIAMOND on the *same* 2000-pair set.

## Citation gaps (most critical)

- **MMseqs2** (Steinegger & Söding, Nat Biotech 2017) — sensitive protein search baseline
- **DIAMOND v2** (Buchfink et al., Nat Methods 2021)
- **CATHe** (Nallapareddy et al., 2023) — remote-homolog baseline on CATH
- **TAPE** (Rao et al., NeurIPS 2019) — standard protein eval
- **ProteinGym** (Notin et al., NeurIPS 2023) — fitness benchmarks
- **Vocab expansion empirical comparison** (arxiv 2407.05841)
- **ProtST** (Xu et al., ICML 2023) — protein + text foundation
- **MoE made intrinsically interpretable** (arxiv 2503.07639)

## Figure concerns

- **Fig 2 (heatmap_delta)**: layer-averaged expert IDs likely invalid. Even ignoring that, expert order arbitrary, no clustering.
- **Fig 3 (per_layer_js)**: strongest figure. Needs bootstrap bands.
- **Fig 4 (js_decomposition)**: undercuts the 98%/2% slogan — clearly shows L28-L29 SFT effects. Caption needs softening.
- **Fig 5 (per_task_gain)**: reads like a slide. No CIs, inconsistent task ordering.
- **fig_per_layer_entropy**: overcrowded, not colorblind-friendly, no CPT panel.
- **fig_delta_key_layers**: separate colorbars per panel, missing panel labels.
- **fig_js_three_way**: should replace Fig 3 in main text.
- All captions: state n=50, 384 tok cap, layer-local vs layer-averaged, color scale shared/independent, what negative values mean.

## Mock review (NeurIPS-style)

**Summary**: Gemma-based bio-language MoE adapted via vocab injection, CPT, and SFT. Argues via router logging that CPT drives representation while SFT drives output alignment.

**Strengths**: Interesting idea (router analysis in bio-MoE). Detailed training recipe. Attempt to separate CPT from SFT. Acknowledges weak remote-homology.

**Weaknesses**:
- Likely fatal: layer-averaged expert-ID analyses
- Author owns the benchmark, no external validation
- Baseline definition unclear/unfair
- Strongest claims unsupported by stats/replication/causal controls
- Hard benchmark does not beat baseline
- 19% purity called "single-function" is a stretch
- Writing overconfident, occasionally misleading

**Score**: 3/10  **Confidence**: 4/5

**Reviewer's parting advice**: "Don't try to polish this into shape. First fix the layer-identity issue, rerun the routing analysis with layer-local experts only, add the tokenizer/format/external-benchmark controls, and cut the headline claims by at least half."

---

## Action plan

| P | Task | Effort | GPU |
|---|---|---|---|
| **P0-1** | Fix abstract Remote 0 → -0.5 | 5 min | no |
| **P0-2** | Layer-local specialty (re-run from .npz) | 30 min | no |
| **P0-3** | Soften atomic-expert claims | 30 min | no |
| **P1-1** | Clarify baseline definition | 15 min | no |
| **P1-2** | Add MMseqs2/DIAMOND/TAPE/ProteinGym/ProtST/vocab refs | 15 min | no |
| **P1-3** | Soften "98% vs 2%" rhetoric | 30 min | no |
| **P2-1** | Prompt-bootstrap CIs | 1 h | no |
| P2-2 | Format-matched routing control | 2-3 h | yes |
| P2-3 | ESM-2 head-to-head on same 2000 set | 1-2 h | yes |
| P3 | Multi-seed CPT/SFT, CATH/SCOP external | 100s GPU-h | yes |
