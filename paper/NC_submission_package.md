# Nature Communications — Submission Package
**OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability**
*Liang Wang, Huazhong University of Science and Technology*

---

## 1. Cover Letter

Dear Editor,

We are pleased to submit our manuscript, *"OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability"*, for consideration as an Article in *Nature Communications*.

The work asks how the post-pretraining of a Mixture-of-Experts language model — continued pretraining (CPT) followed by supervised fine-tuning (SFT) — partitions the labour of acquiring biological capability across the two stages. By installing forward hooks on every router of a Gemma-4-26B-A4B model (30 layers x 128 experts, top-8) and measuring expert-routing distributions across eight task families, we provide what is, to our knowledge, the first quantitative router-level decomposition of CPT vs SFT contributions in a biological foundation model. The headline measurement is a clean 96% / 4% split: under a layer-averaged Jensen-Shannon metric, CPT is responsible for the bulk of cross-task differentiation and SFT contributes a small, layer-localized increment near the language-model head. Bootstrap-resampled 95% confidence intervals exclude zero for both contributions, and a format-matched control retains 90.2% of the signal under a uniform prompt template — ruling out a trivial prompt-formatting explanation.

The same training pipeline produces strong absolute benchmark numbers. On a 500-pair balanced subset of *protein_pair_remote* (BioPAWS), our final model (v5) reaches 82.60% accuracy, outperforming ESM-2 3B by +31.4 percentage points, MMseqs2 by +28.2 pp, and DIAMOND by +29.4 pp on the identical pair set. Notably, scaling ESM-2 from 650M to 3B parameters changes accuracy by only +0.7 pp, indicating that the bottleneck is not encoder capacity but rather the absence of explicit homology supervision during pretraining. On standard homology and BixBench knowledge questions v5 reaches 99.40% and 93.66%, a +14.4-point and +6.7-point improvement over the vocabulary-extended Gemma-4-Instruct baseline. A dual-head architecture introduced in v5 — a generation head plus two per-residue classification heads (Foldseek 3Di, DSSP) trained jointly under a 0.5 / 0.5 loss split — additionally reaches 78.6% per-residue 3Di accuracy (chance 5%) and 100% DSSP accuracy (chance 12.5%) on a held-out 50-protein evaluation, while preserving full chat-style generation capability.

Three aspects of the work are, in our view, of broad interest to the *Nature Communications* readership.

First, the **mechanistic interpretability angle**. MoE routers, unlike attention or hidden-state probes, expose a *discrete, causally-upstream* observable for which subnetwork processes which input. We use this to give a non-trivial answer to a question — "what does CPT do that SFT cannot?" — that has been actively debated for general LLMs but rarely measured directly. The methodology generalizes beyond biology to any MoE foundation model.

Second, the **avoidance of catastrophic forgetting** through a population-displacement mechanism specific to MoE: natural-language tokens are routed to experts that biological tokens do not displace, so the two populations occupy disjoint corners of expert space. In our prior dense-LLM experiments, the same SFT recipe lost 17 points on BixBench; in v3 it gains 6.7 points. The router data make this mechanism visible.

Third, the **release of all model artefacts**. Six model variants — three full BF16 (~50GB each) plus three LoRA + classification-head bundles — are publicly available on Hugging Face under `dnagpt/`. The complete training code is on GitHub (`maris205/omnigene4`). All routing-count matrices and bootstrap samples are in the supplementary archive. We believe this level of openness is unusual for foundation-model work at this scale, and it directly supports the kind of independent replication that the field needs.

We have already deposited a preliminary version on bioRxiv (DOI: 10.1101/2026.01.03.697478) but believe the *Nature Communications* readership — spanning ML researchers, structural biologists, and computational genomicists — is the natural audience for the joint methodological-and-empirical contribution. The work has not been submitted elsewhere.

Sincerely,
**Liang Wang**
School of Artificial Intelligence and Automation
Huazhong University of Science and Technology
Wuhan 430074, China
Email: wangliang.f@gmail.com

---

## 2. Editorial Importance Statement (60 words)

We provide the first router-level decomposition of how continued pretraining and supervised fine-tuning each contribute to biological-foundation-model capability, finding a clean 96% / 4% split in cross-task expert specialization. The same training pipeline yields a +31.4-point gain over ESM-2 3B in remote homology, with all six model variants released openly to the community.

*[Word count: 60]*

---

## 3. Highlights (5 bullets, ~85 chars each)

- Router-level CPT/SFT decomposition: 96% of cross-task expert differentiation acquired during CPT.
- Remote homology +31.4 pp over ESM-2 (3B), +28.2 pp over MMseqs2 on identical 500-pair sample.
- Dual-head architecture: chat generation + per-residue 3Di/DSSP classifiers trained jointly.
- MoE routing reveals catastrophic-forgetting avoidance via expert-population displacement.
- All six model variants (~50GB each, plus LoRA bundles) released openly on Hugging Face.

---

## 4. Recommended Reviewers

### 4.1 Strongly recommended (4)

**1. Prof. Bonnie Berger** — Massachusetts Institute of Technology
Email: bab@mit.edu
*Justification:* Pioneering work on protein structure prediction and homology detection (TM-Vec, PLMSearch). Extensive experience evaluating sequence-search baselines in remote-homology settings; ideal for assessing our +32.1pp ESM-2 gap claim.

**2. Prof. Burkhard Rost** — Technical University of Munich
Email: assistant@rostlab.org
*Justification:* Long-standing authority on protein language models (ProtT5, ESM family analysis) and DSSP secondary-structure prediction. Directly evaluates our 100% DSSP per-residue head claim and 3Di classification-head methodology.

**3. Dr. Sergey Ovchinnikov** — Massachusetts Institute of Technology
Email: so3@mit.edu
*Justification:* Recognized expertise in protein structure-as-language and MSA-based foundation models. Familiar with both Foldseek 3Di and unified DNA/protein foundation modeling — the methodological territory of OmniGene-4.

**4. Prof. Mohammed AlQuraishi** — Columbia University
Email: ma4129@columbia.edu
*Justification:* Pioneer of end-to-end differentiable protein folding and a vocal critic of opaque foundation-model claims. The router-level interpretability angle is exactly the kind of mechanistic claim he typically evaluates rigorously.

### 4.2 Optional alternates (2)

**5. Prof. Yang Zhang** — University of Michigan / NUS
Email: zhng@umich.edu
*Justification:* Authority on protein structure benchmarking (TM-score, I-TASSER) and remote homology evaluation protocols.

**6. Dr. Pascal Notin** — University of Oxford
Email: pascal.notin@stats.ox.ac.uk
*Justification:* Lead developer of ProteinGym and authority on benchmark design for protein language models.

### 4.3 Non-preferred reviewers

- *Open conflict-of-interest declaration:* The author has had no advisory or commercial relationship with any of the recommended reviewers above. No reviewers are non-preferred for personal-conflict reasons.

---

## 5. Data and Code Availability

| Resource | URL |
|---|---|
| Source code | `https://github.com/maris205/omnigene4` |
| bioRxiv preprint | `https://doi.org/10.1101/2026.01.03.697478` |
| Model — v5 merged BF16 (52 GB) | `https://huggingface.co/dnagpt/OmniGene-4-SFT-v5-merged` |
| Model — v5 LoRA + classification heads | `https://huggingface.co/dnagpt/OmniGene-4-SFT-v5` |
| Model — v4 LoRA | `https://huggingface.co/dnagpt/OmniGene-4-SFT-v4` |
| Model — v3 merged BF16 | `https://huggingface.co/dnagpt/OmniGene-4-SFT-v3-merged` |
| Model — v3 GGUF Q4_K_M (16 GB) | `https://huggingface.co/dnagpt/OmniGene-4-SFT-v3-GGUF` |
| Model — CPT-only checkpoint | `https://huggingface.co/dnagpt/OmniGene-4-CPT-v2-merged` |
| Dataset — BioPAWS | `https://huggingface.co/datasets/dnagpt/biopaws` |
| Routing matrices and bootstrap samples | Supplementary `outputs/moe_analysis/` |

---

## 6. Conflict of Interest Statement

The author declares no competing financial or non-financial interests.

---

## 7. Funding Statement

This research was supported by GPU compute donated by io.net (~160 GPU-hours). No additional grant funding was received for this work.

---

## 8. Author Contributions

L.W. conceived the study, designed the architecture and training pipeline, executed all training and evaluation, performed the router-level analysis, drafted and revised the manuscript, and prepared all data and code releases.
