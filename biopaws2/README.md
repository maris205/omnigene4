# BioPAWS-2: A Unified Instruction-Tuning Benchmark Suite for Biological Foundation Models

> **One-line:** BioPAWS-2 reformulates the entire landscape of biological sequence
> classification / regression / retrieval into a single **instruction-tuning QA format**,
> so that *any* model — a specialized PLM with a custom head, a general-purpose LLM
> answering zero-shot, or an LLM fine-tuned on our own training split — can be placed on
> **the same evaluation axis**.

This is the successor to **BioPAWS-1** (the benchmark formalized in *"Emergence of
Biological Structural Discovery in General-Purpose Language Models"*, bioRxiv
`10.64898/2026.01.03.697478`). BioPAWS-1 contained 5 pairwise/single-sequence tasks and
was used as a *probe* of cross-modal syntactic transfer. BioPAWS-2 generalizes that probe
into a **broad-coverage, instruction-tuning-native benchmark + training resource**.

---

## Why BioPAWS-2 is different from every existing bio benchmark

| Benchmark | Format | Zero-shot LLM ready | SFT-trainable | Multimodal | Numeric→QA | Cross-modal (DNA+Prot) |
|---|---|---|---|---|---|---|
| TAPE | PLM + head | ✗ | ✗ | ✗ | ✗ | ✗ |
| PEER | PLM + head | ✗ | ✗ | ✗ | ✗ | ✗ |
| ProteinGym | DMS scores | ✗ | ✗ | ✗ | ✗ | ✗ |
| FLIP | PLM + head | ✗ | ✗ | ✗ | ✗ | ✗ |
| GUE (DNABERT-2) | classification | ✗ | ✗ | ✗ | ✗ | ✗ |
| BixBench | text QA | ✓ | ✗ | ✗ | ✗ | ✗ |
| **BioPAWS-2** | **instruction QA** | **✓** | **✓** | **✓** | **✓** | **✓** |

The gap BioPAWS-2 occupies: **no existing benchmark is simultaneously instruction-tuning
native, SFT-trainable, multimodal, and cross-modal.** Traditional benchmarks force a PLM +
classification-head training loop; BioPAWS-2 turns the *same underlying labels* into
natural-language QA, so a generalist model is evaluated head-to-head with a specialist.

---

## Dual-mode evaluation protocol (the headline)

Every task ships `train / val / test` splits. Each model is scored **twice**:

- **Mode A — Zero-shot QA**: evaluate directly on `test`. Measures instruction-following +
  innate biological prior.
- **Mode B — SFT-then-eval**: LoRA fine-tune on `train`, then evaluate on `test`. Measures
  **trainability** — how much the model improves when given our QA corpus.

Leaderboard reports `base_acc · ft_acc · Δ(ft − base)`. **Δ is a new axis no other
benchmark reports**: it directly quantifies a model's *fine-tunability* on biological QA.
This also dissolves the "self-benchmarking" concern — BioPAWS-2 is a public training
resource that anyone can SFT on; OmniGene-4-MM is merely one of many models scored on it.

---

## Canonical sample schema (chat / `messages` format)

Identical to the OmniGene-4-MM unified corpus, so existing training/eval scripts
(`omnigene5/scripts/*`) consume it unchanged.

```json
{
  "id": "f2_ec_number:00012",
  "task_family": "F2_functional",
  "task_id": "ec_number",
  "modality": ["protein", "text"],
  "images": [],
  "messages": [
    {"role": "user", "content": "Predict the top-level Enzyme Commission (EC) class for this protein.\nSequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ..."},
    {"role": "assistant", "content": "EC 3 (Hydrolase). The conserved S/H/D catalytic triad indicates hydrolytic peptide-bond cleavage."}
  ],
  "answer_short": "3",
  "choices": ["1","2","3","4","5","6","7"],
  "metric": "accuracy",
  "split": "train",
  "license": "CC-BY-4.0",
  "source": "UniProt:P00761"
}
```

Field notes:
- `messages` — chat turns (drives both zero-shot inference and SFT). Multi-image / mixed
  modality supported via `images` + `<image>` placeholders in content.
- `answer_short` — canonical answer for exact-match / numeric scoring.
- `choices` — present for multiple-choice tasks (enables logit-based scoring for PLMs too).
- `metric` — one of `accuracy | f1 | mcc | spearman | exact_match | rouge_l | bleu`.
- `modality` ⊆ `{dna, protein, structure, smiles, vision, text}`.

---

## Task families (9 families, broad coverage)

Design principle: **representativeness + completeness**, modeled on how modern LLM
benchmarks (MMLU, HELM, BIG-bench, GPQA) achieve breadth. Regression/numeric tasks are
**bucketized into ordinal QA** (e.g. high / medium / low) so they fit the QA format while
preserving a Spearman correlation back-channel.

### F1 — Pairwise alignment & homology *(BioPAWS-1 core, kept)*
- protein homology (standard / remote <25% ID), DNA homology, **Central Dogma** (CDS↔protein)
- Source: existing `dnagpt/biopaws` splits.

### F2 — Protein single-sequence functional
- EC number (7-class), GO BP/MF/CC, subcellular localization (DeepLoc), signal peptide,
  transmembrane topology, **solubility (bucketized)**, **thermostability Tm (bucket: high/med/low)**.
- Source: UniProt / SwissProt / DeepLoc / Meltome Atlas.

### F3 — DNA / genomic classification *(broad DNA coverage)*
- promoter detection, enhancer, splice site (donor/acceptor), TF binding site,
  core promoter strength (**bucketized**), histone marks, **5'UTR ribosome load (bucketized)**,
  chromatin accessibility.
- Source: GUE (DNABERT-2), DeepSEA, Basenji, MPRA, Genomic Benchmarks.

### F4 — Variant / mutation effect *(numeric → QA)*
- ProteinGym DMS fitness (**bucketized to beneficial/neutral/deleterious**),
  ClinVar pathogenicity (binary QA), stability ΔΔG (**bucketized**), MAVE.
- Source: ProteinGym v1.1 / ClinVar / MegaScale ΔΔG.

### F5 — Structure-as-text
- SCOPe fold/superfamily/family classification, 3-state & 8-state secondary structure,
  Foldseek 3Di mapping, contact-density (**bucketized**).
- Source: SCOPe 2.08 / DSSP / CATH / Foldseek.

### F6 — Cross-modal (DNA + protein + small molecule mixed) *(user-requested)*
- **DNA↔protein joint** (does this CDS encode this protein? coding-frame consistency),
  protein–drug binding affinity (DTI, **bucketized Kd**), PPI (binary),
  protein → free-form function description, SMILES → physicochem property (**bucketized**).
- Source: DAVIS / BindingDB / STRING / UniProt comments / MoleculeNet.

### F7 — Biological chain-of-thought reasoning
- mental-folding CoT (Helix-Turn-Helix vs TIM-barrel discrimination, from BioPAWS-1
  Qwen-3 traces), multi-step homology justification, mechanism-of-action reasoning.
- Source: curated Qwen-3 / Gemini CoT logs + BioReason data.

### F8 — Biomedical sequence-grounded QA *(user-requested, BixBench-style)*
- T/F + multiple-choice biology QA where the answer depends on an embedded DNA/protein
  sequence (not pure literature recall), gene/variant clinical interpretation,
  pathway membership, disease-gene association.
- Source: BixBench-style construction + PubMedQA(seq-grounded subset) + OMIM.

### F9 — Multimodal QA (image + text) *(paper's new contribution)*
- molecular-structure-image recognition / captioning (Vis-CheBI20),
  protein-structure-cartoon → fold name, microscopy (HPA) → subcellular compartment,
  scientific-figure / chart QA, image→SMILES translation.
- Source: reuse OmniGene-4-MM unified corpus (B_chebi20, C_hpa_microscopy, E_chartqa, ...).

---

## Target scale

~250K QA samples total (each subtask 3–20K), `train/val/test = 80/10/10`.
Held-out test enforces **no train/test leakage** via sequence-identity clustering
(MMseqs2 ≤ split-specific identity threshold) and PDB/UniProt release-date cutoffs.

---

## Repository layout

```
biopaws2/
├── README.md           # this file (design spec)
├── schema.py           # canonical sample dataclass + validator
├── tasks/              # one converter per subtask → messages QA jsonl
│   ├── f1_pairwise/
│   ├── f2_functional/
│   ├── f3_dna/
│   ├── f4_variant/
│   ├── f5_structure/
│   ├── f6_crossmodal/
│   ├── f7_cot/
│   ├── f8_biomed_qa/
│   └── f9_multimodal/
├── eval/               # dual-mode harness (zero-shot + SFT-then-eval)
│   ├── run_zeroshot.py
│   ├── run_sft_eval.py
│   └── score.py
├── data/               # generated jsonl shards (gitignored, → HF dnagpt/biopaws-2)
├── docs/               # leaderboard spec, license table, datasheet
└── scripts/            # build_all.py, upload_hf.py, leakage_check.py
```

## Hugging Face release
- Dataset: `dnagpt/biopaws-2` (v1 = `dnagpt/biopaws` kept as-is for provenance)
- Leaderboard Space: `dnagpt/biopaws-2-leaderboard`
