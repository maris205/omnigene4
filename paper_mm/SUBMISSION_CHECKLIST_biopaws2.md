# BioPAWS-2 / OmniGene-4 — Submission Checklist (Patterns resubmission)

**Status:** Ready to submit. External mock review (GPT-5.4 xhigh, 6 rounds): **7.1/10, weak
accept / minor-revision territory as a resource paper.** Score trajectory:
Reject → 4 → 6 → 6.8 → **7.1**.

**Target:** *Patterns* (Cell Press), resubmission of PATTERNS-D-26-00307 as a NEW submission
(consolidated per editor Alvarado's invitation).

---

## 1. Manuscript artefacts (all in `paper_mm/`, pushed to GitHub `biopaws2-release`)

| File | What | Status |
|---|---|---|
| `omnigene4_mm.pdf` | Main manuscript, **40 pp** | ✅ compiles clean, 0 undefined refs/cites |
| `omnigene4_mm.tex` | Source (retitled to BioPAWS-2-first) | ✅ |
| `section_biopaws2.tex` | §2 BioPAWS-2 (`\input`) — 2 protocol-pure tables | ✅ |
| `section_leakage.tex` | Leakage audit appendix (before/after MMseqs2) | ✅ |
| `supplementary_nc.pdf/.tex` | Supplementary Info, **18 pp** (+ Note 4: BioPAWS-2 construction/leakage/controls) | ✅ compiles, cross-doc refs resolve on merge |
| `refs.bib` | Bibliography (+ llama-gene, PEER, FLIP) | ✅ |
| `cover_letter_resubmission.pdf/.tex` | 2-pp cover letter to Dr. Alvarado | ✅ updated to final state |
| `REVIEW_biopaws2_gpt54.md` | Full 6-round external review record | ✅ (internal, do not submit) |

**Title:** "OmniGene-4: A Locally-Deployable Bio-Language MoE Model with BioPAWS-2, an Open Benchmark for Biological Foundation Models" (model-first; 7.0-7.1 accept-track)
Biological Foundation Models, with OmniGene-4 as a Case Study"

## 2. Public artefacts (reproducibility — links in manuscript + cover letter)

| Artefact | Location | Status |
|---|---|---|
| BioPAWS-2 dataset | HF `dnagpt/biopaws-2` | ✅ 22 jsonl, 306K, dataset card, stats |
| Leakage-free entity-disjoint splits | in dataset (split field) + audit script | ✅ |
| Code (converters, eval, leakage audit, re-split) | GitHub `maris205/omnigene4` → `biopaws2/` | ✅ pushed |
| Model checkpoints | HF `dnagpt/OmniGene-4-MM-merged` + stage3v3 LoRA | ✅ |

## 3. Reviewer-concern → resolution map

| Original / review concern | Resolution | Evidence |
|---|---|---|
| **Self-benchmarking** (editor's reject reason) | Public benchmark + raw-Gemma-base negative control showing bio-CPT gives NO SFT advantage (used own benchmark against self) | §2 control table (0.769 vs 0.760) |
| Consolidate two papers | Retitled/re-led around BioPAWS-2; OmniGene = case study | Title + abstract |
| "Identity-aware splitting asserted, not shown" | MMseqs2 leakage audit released; entity-disjoint re-split (0.78→0.08) | §leakage appendix + audit script |
| Scores may be memorization | Re-ran all supervised results on clean splits: only −1.4pp | §2 + control caption |
| Mixed/footnoted omnibus table | Split into 2 protocol-pure tables (joint-SFT clean-split \| native-protocol reference) | Tables 1–2 |
| ProteinGym bucketized not native | Native Spearman ρ=0.36 added as primary metric | Table 2 + F4 text |
| Router "interpretability" overstated | Already hedged (hypothesis-generating, not mechanistic) | §4 wording |
| Thin baselines / no non-OmniGene FT | Raw-Gemma-base joint-SFT control added (both leaky+clean) | §2 control |

## 4. Known limitations stated in-paper (not blockers for submission)

- No DNA-specialist baseline (DNABERT-2 toolchain incompatibility, documented).
- ESM-2 head reference numbers on pre-leakage-fix splits — labeled as optimistic upper
  bounds in Table 2 caption.
- Central-Dogma residual cross-modal leakage 0.46 (single-tower clustering limit) — flagged.

## 5. Path beyond 7.1 (future work — requires new experiments, NOT needed to submit)

Per reviewer: **7.5** = one orthogonal clean-split baseline (ideally a DNA-specialist model)
turning "model-agnostic resource" from plausible to demonstrated. **~8** = DNA-PLM baseline +
leakage-fixed reruns of native-protocol reference baselines + cross-backbone validation +
causal router validation. All require GPU experiments; none are text edits.

## 6. Pre-submission actions

- [ ] Create PR `biopaws2-release` → main on GitHub (web; `gh` CLI not installed): 
      `github.com/maris205/omnigene4/pull/new/biopaws2-release`
- [ ] Submit `omnigene4_mm.pdf` + `cover_letter_resubmission.pdf` via Patterns portal as a
      new submission, referencing prior PATTERNS-D-26-00307.
- [ ] Confirm HF dataset + repo are public (not gated).
