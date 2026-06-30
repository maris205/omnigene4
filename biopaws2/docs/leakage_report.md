# BioPAWS-2 Leakage / Contamination Audit

Per-task train/val/test leakage. **exact_leak** = fraction of test sequences appearing verbatim in train (entity-level). **homology_leak** = fraction of test-containing MMseqs2 clusters (min-seq-id 0.5, cov 0.5) that also contain a train sequence (subsampled to 20K/split for tractability).

| Task | rows | unique seqs | test exact-leak | MMseqs2 homology-leak |
|---|---|---|---|---|
| f4_proteingym_dms | 27513 | 10 | 0.0 | 0.0 |
| f7_bioreason_cot | 7352 | 7355 | 0.0 | 0.0 |
| f8_bixbench_mcq | 205 | 0 | 0.0 | — |
| f8_bixbench_tf | 205 | 0 | 0.0 | — |
| f9_mol_caption | 3269 | 0 | 0.0 | — |
| f9_mol_iupac | 2680 | 0 | 0.0 | — |
| f9_mol_recog | 3269 | 0 | 0.0 | — |
| int_opi_function | 4473 | 4473 | 0.0 | 0.0 |
| int_protein2text_qa | 2629 | 181 | 1.0 | 1.0 |
| int_protein_catalogue_cot | 29695 | 27996 | 0.0 | 0.0144 |
| int_uniprot_qa | 60000 | 10 | 1.0 | 1.0 |
| lg_central_dogma | 16662 | 26094 | 0.0 | 0.4566 |
| lg_core_promoter_detection | 5920 | 5918 | 0.0 | 0.0 |
| lg_fold_class | 19468 | 19752 | 0.0 | 0.0046 |
| lg_npp | 3364 | 3375 | 0.0 | 0.0 |
| lg_promoter_detection | 21042 | 21018 | 0.0 | 0.0131 |
| lg_signal_peptide | 8304 | 4152 | 0.0 | 0.0051 |
| lg_splice_site | 4562 | 4545 | 0.0 | 0.0 |
| lg_subcellular_loc | 12993 | 11899 | 0.0 | 0.0216 |
| lg_tf_prediction | 3437 | 3437 | 0.0 | 0.0 |
| protein_homology_remote | 25647 | 9062 | 0.0 | 0.0 |
| protein_homology_std | 16168 | 18218 | 0.0 | 0.0845 |
