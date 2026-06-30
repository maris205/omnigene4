# BioPAWS-2 / OmniGene-4-MM — External Peer Review (GPT-5.4, xhigh)

**Date:** 2026-06-28
**Reviewer model:** gpt-5.4 via Codex MCP (xhigh reasoning)
**Thread ID:** `019f12b9-17e0-7240-943c-b1562b6ab636` (resumable)
**Target:** Patterns resubmission `omnigene4_mm.tex` (38pp), focus on new §2 BioPAWS-2.

---

## Verdict

**Reject in current form / at best major-revision.** The revision solves the *appearance*
of self-benchmarking more than the *reality*. The benchmark is still designed, reformulated,
split, scored, and selectively populated by the same group whose model it validates. The
main comparison table uses a best-case, model-specific regime for OmniGene while giving
baselines thinner regimes — reads as "self-benchmarking by another name."

## Single most damaging weakness

> The manuscript cannot distinguish **model quality** from **benchmark-construction
> effects**, and the main table makes it worse by giving OmniGene a model-specific best-case
> regime (joint-SFT for most rows, zero-shot for remote homology because SFT hurts, per-task
> SFT for ProteinGym) while baselines get thinner ones. **The self-benchmarking concern was
> reframed, not resolved.**

## Major findings (condensed)

1. **Public ≠ independent.** HF/GitHub release is procedural, not substantive. Authors still
   control task reformulation, splits, metrics, baseline selection, and which OmniGene
   numbers are shown.
2. **Headline column mixes regimes** → oracle-style comparison, undermines "one model, one
   run, one interface."
3. **Generalist-vs-specialist framing** is legit only as a *secondary* systems argument, not
   a replacement for per-task comparison. 0.583 vs 0.971 on fold is a major loss, not a
   "modest macro gap." "The fair question is not X but Y" is a false choice — invites
   reviewer anger.
4. **Baselines too thin.** Missing **fine-tuned general-LLM baseline** is near-fatal: cannot
   separate "BioPAWS-2 data is good" from "OmniGene is good." "Toolchain incompatibility"
   (DNABERT-2) is not an acceptable excuse in a benchmark paper.
5. **Specialist baselines selectively weak exactly where OmniGene claims wins** (ProteinGym
   mean-pool+head is not community-standard variant-effect eval).
6. **Construction rigor below bar.** "Identity-aware splitting" is an assertion — needs exact
   thresholds, tools, split units, leakage stats, contamination audit vs OmniGene CPT/SFT.
7. **Metrics not defensible as primary.** 3-bucket regression loses signal; ROUGE-L/BLEU weak
   for bio reasoning; F7 CoT 0.082–0.705 wildly.
8. **Reads as two half-papers stapled together.** Benchmark section looks reactive.

## Logical gaps flagged
- "Fair question is not X but Y" = false choice (both fair).
- "One model, one run" contradicted if rows mix zero-shot + per-task SFT.
- "SFT failure on remote homology supports abductive reasoning" = speculative (could be
  imbalance/prompt/optimization).
- "ProteinGym shows pooled PLM loses mutation signal" = too broad without standard baselines.
- "Subsumes TAPE/PEER/FLIP/..." = overstated (QA reformulation ≠ subsuming their paradigms).
- "First / to our knowledge" = risky, soften.

---

## AGREED MINIMAL SURVIVAL PACKAGE (do 1–5 first)

| # | Action | Cost | Lift |
|---|---|---|---|
| 1 | **Re-cut into 3 protocol-clean tables**: zero-shot \| joint-multitask-SFT \| per-task-SFT. Every row populated for ALL applicable models. No mixed headline column. | 0 GPU-d | Very high |
| 2 | **Leakage/contamination appendix** + cross-family overlap matrix (see spec below). Release split manifest, not just converter. | 0–0.5 GPU-d | Very high |
| 3 | **Run `raw Gemma-4-base + identical joint-SFT`** — the single most informative baseline. Same arch/data/recipe/budget, different init. | 2–4 GPU-d | Very high |
| 4 | **Repair F4**: WT-grouped split, native Spearman as primary metric, ≥1 stronger native baseline. | 0.5–2 GPU-d | High |
| 5 | **Remove F7 from primary macro** unless scored better than ROUGE/BLEU. | 0 GPU-d | High |
| 6 | (if Gemma ambiguous) second joint-SFT generalist baseline | 2–4 GPU-d | Med |
| 7 | (to claim CPT specifically) `vocab-only, no-CPT + same SFT` ablation | 2–4 GPU-d | Med |
| 8 | DNA specialist baseline (only if cheap) | 1–3 GPU-d | Med-low |

### Repositioning the claims
- **Primary model claim:** OmniGene vs raw Gemma-base under identical joint-SFT.
- **Primary benchmark claim:** BioPAWS-2 = common instruction-format + split framework with
  explicit leakage controls.
- **Secondary systems claim:** specialists still win some tasks; generalist tradeoff is
  deployment breadth, not universal superiority.
- **Stop saying** "the fair question is not X but Y." Report both.

---

## LEAKAGE / RIGOR APPENDIX SPEC

**Global per-subtask table (all 22):** source+version, raw count, canonical entity key,
split unit, dedup tool, dedup threshold, entity-level train/val/test, QA-row-level
train/val/test, intra-split dups removed, cross-split dups removed, nearest-train overlap
stat, cross-family train exposure rate, contamination rate vs OmniGene CPT, vs OmniGene SFT,
primary metric, secondary metric, limitations.

**Non-negotiable rules:** split BEFORE QA templating; all prompts from same entity → one
split; release split manifest; cross-family overlap matrix (% test entities of family A
appearing in training families B–I); if non-trivial overlap → define `strict entity-disjoint
core` as the primary joint-SFT benchmark.

**Per-family split unit + tool:**
- F1 homology: protein **cluster** (MMseqs2), 0 shared protein/cluster across splits, report
  nearest-train identity dist; explicitly audit the <25% remote boundary.
- F2 function: UniProt accession / cluster (MMseqs2); cross-overlap with F5/F6.
- F3 DNA: genomic **locus** (BEDTools interval overlap) + nucleotide dedup + reverse-complement
  canonicalization; 0 overlapping loci; same-locus windows kept together.
- F4 variant: **WT background / accession** grouping — all mutants of one WT in one split;
  cluster WTs (MMseqs2); report % test mutants whose WT (or close homolog) is in train (→0);
  **primary metric = Spearman on raw DMS**, buckets from train only.
- F5 structure: accession/cluster; family/superfamily leakage; native classification metric
  primary.
- F6 cross-modal: split at **gene/compound level** (underlying entity, not view).
- F7 CoT: source problem/document; ROUGE/BLEU auxiliary only; move to exploratory OR add
  exact-answer-extraction / rubric scoring on a subset.
- F8 biomed QA: source passage + accession; keep templated variants together; report
  source-doc overlap + text near-dup.
- F9 multimodal: canonical molecule (InChIKey/SMILES); perceptual-hash image near-dup; same
  molecule not in different splits via different renderings.

**Contamination vs OmniGene CPT/SFT (dedicated subsection):** per family report exact +
near-dup overlap vs CPT and vs SFT. Tools: proteins MMseqs2, DNA hash+nucleotide cluster,
text MinHash/n-gram Jaccard, molecules SMILES/InChIKey, images perceptual hash. **State
explicitly what cannot be audited** (opaque external-LLM pretraining, vision-tower
pretraining) — honesty helps more than false global rule-out.

---

## RESULTS-TO-CLAIMS MATRIX (Gemma-base + same SFT vs OmniGene + same SFT)

Definitions: `≈` = macro diff within CI / tiny / sign inconsistent across families;
`>>` = clear gap, consistent sign across multiple families (not one cherry-picked task).

| Outcome | Claim ALLOWED | Claim FORBIDDEN | Editorial consequence |
|---|---|---|---|
| **Gemma+SFT ≈ OmniGene+SFT** | BioPAWS-2 is a useful multi-task SFT resource; most gain from supervised adaptation, not OmniGene init | OmniGene's bio-CPT is the main reason for joint-SFT perf | Reposition as **benchmark-first** paper; OmniGene = one case study |
| **OmniGene+SFT >> Gemma+SFT** | Under matched arch + identical SFT, OmniGene init improves downstream joint multi-task perf | Cannot claim CPT *alone* unless vocab-only ablation run (tokenizer differs) | Best case for combined paper (if leakage appendix clean) |
| **Gemma+SFT >> OmniGene+SFT** | BioPAWS-2 shows domain CPT does NOT auto-improve joint QA; OmniGene may help some zero-shot but hurt joint SFT | OmniGene is a stronger generalist on this benchmark | Combined model-paper very hard; pivot to benchmark-first or split |

**Extra branch rules:**
- OmniGene wins only F1, loses other families → claim is `task-specific strength`, NOT
  `modality-invariant transfer`.
- OmniGene wins overall but only via one heavily-weighted family → report family-wise deltas,
  don't sell broad superiority.

---

## TWO STRATEGIC PATHS (reviewer's framing)
1. **Benchmark-first:** make BioPAWS-2 a real first-class benchmark paper — strong external
   baselines, rigorous leakage controls, clean protocols, OmniGene as one case study.
2. **Model-first on independent benchmarks:** stop selling BioPAWS-2 as the fairness
   solution; validate OmniGene primarily on third-party benchmarks.

"Right now it tries to do both and is convincing at neither."

---

## DECISION / NEXT ACTIONS
The honest read: the current celebratory framing ("chat reverses PLM", "generalist wins")
overclaims. The path forward is the 5-item package — most of it is **rewrite + 1 GPU run
(Gemma-base control) + leakage appendix**, all feasible on current hardware. The
results-to-claims matrix is pre-committed so we report honestly whatever the Gemma control
shows.

---

## ROUND 3 — Mock Review WITH SCORES (post Gemma-base control + benchmark-first revision)
**Date:** 2026-06-30 | **New thread:** `019f1660-020b-7531-9d45-52d703a25599`

### Score: **4/10 · Major revision · confidence 4/5**

**Verdict shift:** "No longer a principled reject on fairness grounds. The negative control
genuinely improves the paper." → We escaped the auto-reject zone. But 4/10 is still far from
accept.

### What worked
- The `Gemma-base + identical SFT` negative control is "the strongest addition" — convincing
  precisely because it weakens our own story (honesty over flattery).
- Public BioPAWS-2 resource, dual-mode protocol, base/ft/Δ reporting all praised.
- Zero-shot remote homology = clearest model-specific reason to care about OmniGene.
- Router CPT-vs-SFT decomposition "more substantial than cosmetic."
- Over-claim softening acknowledged.

### The NEW core problem (center of gravity)
The control data say BioPAWS-2 is the stronger supervised contribution, but the title/framing
still sell an OmniGene paper. "Four papers compressed into one." Hierarchy unresolved.

### Residual reject risks (still substantial)
1. Benchmark not yet audit-proof: "identity-aware splitting" asserted, no task-wise
   thresholds/contamination audit. THE remaining credibility bottleneck — and the zero-shot
   remote-homology claim (our best) is the MOST leakage-sensitive, needs cleanest split docs.
2. Results presentation still mixed (footnote-heavy omnibus table) → need 3 protocol-separated tables.
3. Baseline panel still thin → need ≥1 more matched fine-tuned non-OmniGene LLM beyond Gemma.
4. ProteinGym should use native Spearman primary metric, not bucketized accuracy.
5. Router "interpretability" overstated — shows WHEN differentiation arises, not what it MEANS / causal use.
6. No DNA specialist baseline.

### Highest-leverage change toward accept
**Recast explicitly as a BioPAWS-2 benchmark/resource paper** + make benchmark unimpeachable:
protocol-separated tables, field-standard primary metrics, full task-wise split/leakage
documentation, ≥1 additional matched non-OmniGene LLM fine-tuning baseline. Without this
package → another reject likely.

### Questions for authors (to resolve before resubmission)
- What IS the primary paper: BioPAWS-2 / OmniGene-4 / router interp / multimodal? Pick one hierarchy.
- Willing to retitle around BioPAWS-2, OmniGene as case study?
- Add ≥1 matched fine-tuned open LLM baseline beyond Gemma?
- Full leakage appendix (split rules, homology thresholds, dedup, overlap-vs-CPT audit)?
- ProteinGym native metric in main text?
- Router result: causal/semantic validation beyond divergence accounting?
