# BioPAWS-2 Coverage Matrix

**The thesis:** BioPAWS-2 re-expresses the *entire* traditional bio-sequence evaluation
landscape as instruction-tuning QA. The columns below show, for each subtask, **which
traditional benchmark / dataset it subsumes** and **how a numeric or classification label
becomes a QA item**. Completeness is the product spec: if a capability is tested anywhere
in the PLM / genomics / bio-LLM literature, it should have a BioPAWS-2 QA analogue.

Conversion legend:
- **CLS** = native classification → multiple-choice QA (`choices`, metric=accuracy/f1/mcc)
- **NUM→ord** = regression/numeric → ordinal bucket QA (low/med/high, metric=spearman on `meta.value`)
- **GEN** = free-form generation (metric=rouge_l/bleu)
- **PAIR** = pairwise comparison QA (binary)

| Family | Subtask | Conv | Traditional source it subsumes | Modality |
|---|---|---|---|---|
| **F1 Pairwise** | protein homology (standard) | PAIR | BioPAWS-1 / SCOPe pairs | protein |
| | protein homology (remote <25% ID) | PAIR | BioPAWS-1 remote / SCOPe superfamily | protein |
| | DNA homology | PAIR | BioPAWS-1 DNA | dna |
| | Central Dogma (CDS↔protein) | PAIR | BioPAWS-1 cross-modal | dna+protein |
| **F2 Functional** | EC number (top-class + full) | CLS | DeepEC / CLEAN / ProtT5-EC | protein |
| | GO BP / MF / CC | CLS | DeepGOPlus / CAFA | protein |
| | subcellular localization (10-way) | CLS | DeepLoc 2.0 | protein |
| | signal peptide | CLS | SignalP 6.0 | protein |
| | transmembrane topology | CLS | TMbed / DeepTMHMM | protein |
| | solubility | NUM→ord | DeepSol / PROSO II | protein |
| | thermostability Tm | NUM→ord | Meltome Atlas (FLIP) | protein |
| | fluorescence brightness | NUM→ord | TAPE Fluorescence | protein |
| | stability landscape | NUM→ord | TAPE Stability | protein |
| **F3 DNA** | promoter detection | CLS | GUE / DeepSEA / Genomic Benchmarks | dna |
| | enhancer (human/mouse) | CLS | GUE / EnhancerPred | dna |
| | splice site (donor/acceptor) | CLS | SpliceAI / GUE | dna |
| | TF binding site | CLS | GUE 690 ENCODE / DeepSEA | dna |
| | core promoter strength | NUM→ord | MPRA / Basenji | dna |
| | histone marks (H3K4me3 etc.) | CLS | GUE EMP | dna |
| | 5'UTR ribosome load (MRL) | NUM→ord | Optimus 5-Prime MPRA | dna |
| | chromatin accessibility | CLS | Basenji / Enformer ATAC | dna |
| | species classification (DNA) | CLS | Genomic Benchmarks | dna |
| **F4 Variant** | DMS fitness effect | NUM→ord | ProteinGym v1.1 (217 assays) | protein |
| | clinical pathogenicity | CLS | ClinVar / AlphaMissense set | protein |
| | stability ΔΔG | NUM→ord | MegaScale / FireProtDB | protein |
| | non-coding variant effect | CLS | GPN / Enformer-eQTL | dna |
| **F5 Structure** | SCOPe fold | CLS | TAPE Remote Homology / SCOPe | protein+structure |
| | SCOPe superfamily / family | CLS | SCOPe 2.08 | protein+structure |
| | 3-state secondary structure | CLS | TAPE SS-Q3 / NetSurfP | protein |
| | 8-state secondary structure | CLS | DSSP / NetSurfP-3 | protein |
| | Foldseek 3Di mapping | GEN | Foldseek 3Di alphabet | protein+structure |
| | contact density | NUM→ord | ProteinNet contact map | protein |
| **F6 Crossmodal** | CDS encodes protein? (frame) | PAIR | central-dogma consistency | dna+protein |
| | protein–drug binding (DTI) | NUM→ord | DAVIS / KIBA / BindingDB Kd | protein+smiles |
| | protein–protein interaction | CLS | STRING / PEER PPI | protein |
| | protein → function description | GEN | UniProt comments / Prot2Text | protein+text |
| | SMILES → physicochem property | NUM→ord | MoleculeNet (ESOL/Lipo/FreeSolv) | smiles |
| | drug–target activity class | CLS | ChEMBL activity | protein+smiles |
| **F7 CoT** | mental-folding (motif discrim.) | GEN | BioPAWS-1 Qwen-3 traces | protein |
| | multi-step homology justification | GEN | curated CoT | protein |
| | mechanism-of-action reasoning | GEN | BioReason data | protein+text |
| **F8 Biomed QA** | sequence-grounded T/F biology | CLS | BixBench-style | protein/dna+text |
| | gene/variant clinical interpret. | CLS | OMIM / ClinVar narrative | protein+text |
| | pathway membership | CLS | KEGG / Reactome | protein+text |
| | disease–gene association | CLS | DisGeNET | dna+text |
| **F9 Multimodal** | molecule image recognition | CLS | Vis-CheBI20 struct_recog | vision+text |
| | molecule image captioning | GEN | Vis-CheBI20 struct_cap | vision+text |
| | image → SMILES | GEN | OCSR / DECIMER | vision+smiles |
| | protein cartoon → fold name | CLS | rendered PDB/AlphaFold | vision+text |
| | microscopy → subcellular loc. | CLS | HPA immunofluorescence | vision+text |
| | scientific figure / chart QA | CLS/GEN | ChartQA / SciCap | vision+text |

**Totals:** 9 families · ~45 subtasks · covers TAPE, PEER, FLIP, ProteinGym, GUE,
Genomic Benchmarks, DeepLoc, SignalP, MoleculeNet, DAVIS/KIBA, SCOPe, BixBench,
Vis-CheBI20, HPA, ChartQA — in **one instruction-tuning format**.

## Extensibility contract

Adding a capability never requires touching the schema. A contributor:
1. writes `tasks/<family>/<subtask>.py` that yields dicts passing `schema.validate()`,
2. registers `(family, subtask, source, license, metric)` in `tasks/registry.py`,
3. runs `scripts/leakage_check.py` (MMseqs2 identity split for sequence tasks).

This is the "good extensibility" property: BioPAWS-2 is a *protocol*, not a frozen set —
new assays, new modalities, new organisms drop in as long as they emit valid QA records.

## How this differs from everything else (one sentence each)
- **vs TAPE/PEER/FLIP/GUE**: those are (sequence,label) rows requiring a trained head;
  ours are chat records a generalist answers or fine-tunes on directly.
- **vs ProteinGym**: regression scores → ordinal QA, so an LLM competes with a regressor.
- **vs BixBench**: ours is sequence-grounded (answer depends on the embedded sequence) and
  SFT-trainable, not literature recall.
- **vs MMLU/HELM (general LLM benches)**: same breadth philosophy, but every item is a
  *biological* capability with a verifiable gold answer and a dual zero-shot/SFT protocol.
