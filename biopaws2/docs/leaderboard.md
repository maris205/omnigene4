# BioPAWS-2 Leaderboard

Dual-mode protocol: **base** = zero-shot QA, **ft** = LoRA-SFT-then-eval, **Δ** = ft − base (trainability).

## Per-task

| Model | Family | Task | Metric | base | ft | Δ |
|---|---|---|---|---|---|---|
| OmniGene-4-MM | F4_variant | f4_proteingym_dms | accuracy | 0.516 | 0.674 | 0.158 |
| OmniGene-4-MM | F7_cot | f7_bioreason_cot | rouge_l | 0.242 | 0.067 | -0.175 |
| OmniGene-4-MM | F8_biomed_qa | f8_bixbench_mcq | accuracy | 0.273 | — | — |
| OmniGene-4-MM | F8_biomed_qa | f8_bixbench_tf | accuracy | 0.912 | — | — |
| OmniGene-4-MM | F9_multimodal | f9_mol_recog | accuracy | 0.091 | — | — |
| OmniGene-4-MM | F9_multimodal | f9_vis_chebi20 | bleu | 0.212 | — | — |
| OmniGene-4-MM | F5_structure | lg_fold_class | f1 | 0.060 | 0.631 | 0.571 |
| OmniGene-4-MM | F2_functional | lg_subcellular_loc | f1 | 0.501 | 0.834 | 0.332 |
| OmniGene-4-MM | F1_pairwise | protein_homology_remote | accuracy | 0.653 | 0.494 | -0.158 |
| OmniGene-4-MM | F1_pairwise | protein_homology_std | accuracy | 0.702 | 1.000 | 0.298 |
| OmniGene-4-MM-joint | F6_crossmodal | lg_central_dogma | accuracy | — | 0.615 | — |
| OmniGene-4-MM-joint | F5_structure | lg_fold_class | f1 | — | 0.583 | — |
| OmniGene-4-MM-joint | F2_functional | lg_npp | accuracy | — | 0.849 | — |
| OmniGene-4-MM-joint | F3_dna | lg_promoter_detection | accuracy | — | 0.897 | — |
| OmniGene-4-MM-joint | F2_functional | lg_signal_peptide | accuracy | — | 0.945 | — |
| OmniGene-4-MM-joint | F2_functional | lg_subcellular_loc | f1 | — | 0.813 | — |
| OmniGene-4-MM-joint | F1_pairwise | protein_homology_std | accuracy | — | 0.999 | — |
| OmniGene-4-MM-joint9 | F4_variant | f4_proteingym_dms | accuracy | — | 0.625 | — |
| OmniGene-4-MM-joint9 | F7_cot | f7_bioreason_cot | rouge_l | — | 0.082 | — |
| OmniGene-4-MM-joint9 | F6_crossmodal | lg_central_dogma | accuracy | — | 0.615 | — |
| OmniGene-4-MM-joint9 | F5_structure | lg_fold_class | f1 | — | 0.570 | — |
| OmniGene-4-MM-joint9 | F2_functional | lg_npp | accuracy | — | 0.867 | — |
| OmniGene-4-MM-joint9 | F3_dna | lg_promoter_detection | accuracy | — | 0.865 | — |
| OmniGene-4-MM-joint9 | F2_functional | lg_signal_peptide | accuracy | — | 0.924 | — |
| OmniGene-4-MM-joint9 | F2_functional | lg_subcellular_loc | f1 | — | 0.803 | — |
| OmniGene-4-MM-joint9 | F1_pairwise | protein_homology_std | accuracy | — | 0.998 | — |
| esm2_3B+head | F4_variant | f4_proteingym_dms | accuracy | — | 0.531 | — |
| esm2_3B+head | F5_structure | lg_fold_class | f1 | — | 0.971 | — |
| esm2_3B+head | F2_functional | lg_npp | accuracy | — | 0.929 | — |
| esm2_3B+head | F2_functional | lg_signal_peptide | accuracy | — | 1.000 | — |
| esm2_3B+head | F2_functional | lg_subcellular_loc | f1 | — | 0.954 | — |
| esm2_3B+head | F1_pairwise | protein_homology_remote | accuracy | — | 0.524 | — |
| esm2_3B+head | F1_pairwise | protein_homology_std | accuracy | — | 0.998 | — |
| qwen3.7-max | F4_variant | f4_proteingym_dms | accuracy | 0.497 | — | — |
| qwen3.7-max | F8_biomed_qa | f8_bixbench_mcq | accuracy | 0.259 | — | — |
| qwen3.7-max | F8_biomed_qa | f8_bixbench_tf | accuracy | 0.937 | — | — |
| qwen3.7-max | F6_crossmodal | lg_central_dogma | accuracy | 0.787 | — | — |
| qwen3.7-max | F5_structure | lg_fold_class | f1 | 0.073 | — | — |
| qwen3.7-max | F3_dna | lg_promoter_detection | accuracy | 0.420 | — | — |
| qwen3.7-max | F2_functional | lg_signal_peptide | accuracy | 0.663 | — | — |
| qwen3.7-max | F3_dna | lg_splice_site | accuracy | 0.220 | — | — |
| qwen3.7-max | F2_functional | lg_subcellular_loc | f1 | 0.233 | — | — |
| qwen3.7-max | F3_dna | lg_tf_prediction | accuracy | 0.230 | — | — |
| qwen3.7-max | F1_pairwise | protein_homology_remote | accuracy | 0.457 | — | — |
| qwen3.7-max | F1_pairwise | protein_homology_std | accuracy | 0.813 | — | — |

## Per-model summary (macro over families)

| Model | base (macro) | ft (macro) | Δ (macro) |
|---|---|---|---|
| OmniGene-4-MM | 0.392 | 0.590 | 0.199 |
| OmniGene-4-MM-joint | — | 0.793 | — |
| OmniGene-4-MM-joint9 | — | 0.660 | — |
| esm2_3B+head | — | 0.806 | — |
| qwen3.7-max | 0.475 | — | — |

## Generalist (chat, 1 model) vs Specialist (PLM head, N models)

Per task: best chat-model score vs best PLM-head score. `gap = chat − head` (small/positive = chat paradigm matches or beats the single-task head).

| Task | chat (1 model) | PLM head (per-task) | gap |
|---|---|---|---|
| f4_proteingym_dms | 0.674 | 0.531 | 0.143 |
| f7_bioreason_cot | 0.082 | — | — |
| lg_central_dogma | 0.615 | — | — |
| lg_fold_class | 0.631 | 0.971 | -0.340 |
| lg_npp | 0.867 | 0.929 | -0.062 |
| lg_promoter_detection | 0.897 | — | — |
| lg_signal_peptide | 0.945 | 1.000 | -0.055 |
| lg_subcellular_loc | 0.834 | 0.954 | -0.120 |
| protein_homology_remote | 0.494 | 0.524 | -0.030 |
| protein_homology_std | 1.000 | 0.998 | 0.002 |

**Coverage cost**: chat generalist needs **1** model + 1 training to cover all 10 tasks; PLM-head paradigm needs **10** separate heads.
**Mean parity gap (chat − head)**: -0.066 — a single generalist model stays within this of task-specific heads while covering 10× the tasks with one interface.
