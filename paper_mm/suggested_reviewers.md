# Suggested Reviewers — BioPAWS-2 / OmniGene-4 (Patterns resubmission)

Paste one per line into the "Suggest Reviewers" field; the Reason text goes in the comments
box for each. Tuned to the **benchmark-first** framing (BioPAWS-2 as the central
contribution; OmniGene-4 as a case study). None are collaborators, co-authors, co-funded,
or institutionally affiliated with the author (HUST).

---

## Primary suggestions

**Prof. Burkhard Rost, Technical University of Munich** — assistant@rostlab.org
Reason: Long-standing authority on protein language models (ProtT5, ESM-family analysis),
secondary-structure prediction, and — critically for this submission — the standards a
protein-sequence *benchmark* must meet (homology-aware splitting, leakage control). Ideal
for judging our MMseqs2 leakage audit, entity-disjoint re-splitting, and whether BioPAWS-2's
per-task protocols are defensible.

**Dr. Pascal Notin, University of Oxford / Harvard** — pascal.notin@stats.ox.ac.uk
Reason: Lead author of ProteinGym, the field-standard variant-effect benchmark we
incorporate (F4) and now report with native Spearman. The single most relevant reviewer for
assessing benchmark construction rigor, split integrity, and metric choices — exactly the
axis on which this paper now lives.

**Prof. Bonnie Berger, Massachusetts Institute of Technology** — bab@mit.edu
Reason: Pioneering work on remote-homology detection and sequence search (TM-Vec,
PLMSearch). Directly qualified to evaluate our remote-homology claims and the fairness of
the specialist-PLM vs. generalist-LLM comparison, including whether the ESM-2 baselines are
appropriately configured.

**Dr. Sergey Ovchinnikov, Massachusetts Institute of Technology** — so3@mit.edu
Reason: Expertise in protein structure-as-language, MSA-based foundation models, and
Foldseek 3Di — the methodological territory of the structure-as-text tasks (F5) and the
OmniGene-4 case study. Well placed to judge the cross-modal (DNA+protein) tasks and the
honesty of the bio-CPT negative-control finding.

## Optional alternates (if a primary declines)

**Prof. Mohammed AlQuraishi, Columbia University** — ma4129@columbia.edu
Reason: Vocal critic of opaque foundation-model claims; would rigorously scrutinize the
router-level analysis (§4) and the self-critical framing (bio-CPT confers no SFT advantage),
which is a strength for stress-testing the paper's honesty.

**Prof. Yang Zhang, University of Michigan / NUS** — zhng@umich.edu
Reason: Deep experience in protein structure/function prediction and community benchmarking
(CASP-style evaluation), relevant to whether BioPAWS-2's task coverage and scoring are
representative and reproducible.

---

## Conflict-of-interest note (for the COI field)
All suggested reviewers are independent of the author: no shared institution (author is at
HUST, Wuhan), no co-authorship, no known shared funding or active collaboration. Suggested
purely on topical expertise in protein/DNA foundation models and biological benchmark
construction.

## Change vs. prior (NC) submission
Reordered and re-justified for the benchmark-first framing: **Notin (ProteinGym) added and
elevated** as the benchmark-methodology expert; Rost elevated for benchmark/leakage-standard
judgment; Berger/Ovchinnikov retained; AlQuraishi/Zhang moved to alternates. Reasons rewritten
to point at benchmark rigor (leakage, splits, metrics, fair baselines) rather than only the
model's interpretability angle.
