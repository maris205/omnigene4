#!/bin/bash
# Upload all OmniGene-4 / OmniGene-4-MM training data to HF datasets under dnagpt/.
#
# Pattern: classic HTTP per-file with verify-after-commit retry (hf-xet
# silently dies on long uploads via autodl turbo proxy, so we avoid it).
#
# Usage:
#   HF_TOKEN=hf_xxx ./upload_datasets.sh small   # SFT + MM (~1.2 GB total)
#   HF_TOKEN=hf_xxx ./upload_datasets.sh cpt     # CPT bio + 3Di/SS (~96 GB)
#   HF_TOKEN=hf_xxx ./upload_datasets.sh all     # everything

# NOTE: do NOT use `set -e` here -- the retry logic depends on bash
# continuing past non-zero exit codes from `hf upload`.
source /etc/network_turbo
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"

WHICH="${1:-small}"

# ---------- helpers ----------
hf_create_dataset() {
  local repo="$1"
  python - <<EOF
from huggingface_hub import create_repo
import os
create_repo("$repo", repo_type="dataset", token=os.environ["HF_TOKEN"], exist_ok=True)
print("repo OK")
EOF
}

hf_already_has() {
  local repo="$1" path="$2"
  python - <<EOF
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    fs = api.list_repo_files("$repo", repo_type="dataset")
    print("YES" if "$path" in fs else "NO")
except Exception:
    print("NO")
EOF
}

upload_file_with_retry() {
  local repo="$1" src="$2" dst="$3"
  if [ "$(hf_already_has "$repo" "$dst")" = "YES" ]; then
    echo "    [SKIP] $dst already on HF"
    return
  fi
  echo "    [UPLOAD] $dst ($(du -h "$src" | cut -f1))"
  for attempt in $(seq 1 6); do
    set +e
    hf upload "$repo" "$src" "$dst" --token "$HF_TOKEN" --commit-message "Add $dst" --repo-type dataset
    rc=$?
    set -e
    sleep 5
    if [ "$(hf_already_has "$repo" "$dst")" = "YES" ]; then
      echo "    attempt $attempt OK (rc=$rc)"
      return
    fi
    echo "    attempt $attempt: rc=$rc, on-HF=NO; retrying..."
    sleep $((attempt * 15))
  done
  echo "    GIVING UP on $dst"
  return 1
}

# ============================================================
# Repo 1: omnigene4-sft-data
# ============================================================
upload_sft() {
  REPO="dnagpt/omnigene4-sft-data"
  echo "=== $REPO ==="
  hf_create_dataset "$REPO"

  # README first
  cat > /tmp/sft_readme.md <<'README'
---
license: cc-by-4.0
language:
- en
- zh
size_categories:
- 100K<n<1M
task_categories:
- text-generation
- question-answering
tags:
- biology
- protein
- DNA
- bioinformatics
---

# OmniGene-4 SFT corpus

Supervised fine-tuning data for the OmniGene-4 / OmniGene-4-MM family.
See https://github.com/maris205/omnigene4 for the training scripts that
consume these files.

## Files

| File | Rows | Used by | Description |
|---|---|---|---|
| `bio_sft_v2_train.jsonl` | ~179K | Bio-SFT v2 | Eight task families: protein homology (BioPAWS), DNA, structure (3Di/DSSP), cell biology, molecules, mutation, structure prediction, general bio QA |
| `distill_seed.jsonl` | ~6K | seed-only | Initial distillation seed used to bootstrap the v2 corpus |
| `train/omnigene_sft_v1_train.jsonl` | ~179K | SFT v3 base | Same as v2 with cleaner schema and Alpaca template |
| `train/omnigene_sft_v1_train_with_remote.jsonl` | **~199K** | **SFT v3-v5 (final)** | Above + 20K BioPAWS `protein_pair_remote` rows; this is the file used by Bio-SFT v3, v4, v5, and the OmniGene-4-MM Stage 2/3 LoRA training |
| `eval/omnigene_sft_v1_eval.jsonl` | ~1.5K | held-out eval | Used by `40-eval_omnigene4mm.py`, `60-eval_stage2.py`, `92-eval_stage3v3.py`, etc. |
| `master/omnigene_sft_v1_master.jsonl` | ~285K | (intermediate) | Pre-split master corpus before train/eval split |
| `master/cell_sft_master.jsonl` | ~37K | task subset | Cell-biology SFT subset |
| `master/mol_sft_master.jsonl` | ~50K | task subset | Molecule SFT subset |
| `stats/data_mix_report.json` | — | metadata | Per-category counts and ratios |

## Schema

Each line is a JSON object with at least `instruction`, `input`, `output`,
and `category`. Some rows additionally carry `task_name` and `subtask` for
fine-grained accounting.

## Citation

```bibtex
@article{wang2026omnigene4,
  author    = {Wang, Liang},
  title     = {{OmniGene-4}: A Unified Bio-Language MoE Model with Router-Level
               Interpretability and Modality-Invariant Transfer},
  year      = {2026},
  journal   = {bioRxiv},
  doi       = {10.64898/2026.05.12.724542}
}
```
README
  upload_file_with_retry "$REPO" /tmp/sft_readme.md "README.md"

  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/bio_sft_v2_train.jsonl "bio_sft_v2_train.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/distill_seed.jsonl "distill_seed.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train.jsonl "train/omnigene_sft_v1_train.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train_with_remote.jsonl "train/omnigene_sft_v1_train_with_remote.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/eval/omnigene_sft_v1_eval.jsonl "eval/omnigene_sft_v1_eval.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/master/omnigene_sft_v1_master.jsonl "master/omnigene_sft_v1_master.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/master/cell_sft_master.jsonl "master/cell_sft_master.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/master/mol_sft_master.jsonl "master/mol_sft_master.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-fs/omnigene_v2/sft_data/stats/data_mix_report.json "stats/data_mix_report.json"
  echo "  $REPO done"
}

# ============================================================
# Repo 2: omnigene4-mm-corpus
# ============================================================
upload_mm() {
  REPO="dnagpt/omnigene4-mm-corpus"
  echo "=== $REPO ==="
  hf_create_dataset "$REPO"

  cat > /tmp/mm_readme.md <<'README'
---
license: cc-by-4.0
language:
- en
size_categories:
- 100K<n<1M
task_categories:
- image-text-to-text
- text-generation
tags:
- biology
- multimodal
- vision-language
- protein
- DNA
- bioinformatics
---

# OmniGene-4-MM unified corpus

Multi-modal training corpus used for the OmniGene-4-MM Stages 1–3
(see https://github.com/maris205/omnigene4 ).

Each row is a JSON object with `messages` (chat-format), `images`
(list of relative image paths), and `modality` field. Vision rows
reference images that live in the source datasets:

- Vis-CheBI20 ([PharMolix/Vis-CheBI20](https://huggingface.co/datasets/PharMolix/Vis-CheBI20))
- PubMedVision ([FreedomIntelligence/PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision))
- HPA10M (Human Protein Atlas microscopy)
- ChartQA ([HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA))
- Synthetic biomedical visual tasks (project-internal)

To reproduce the MM training, download the source vision datasets to
matching local paths and run `omnigene5/scripts/02-fix_image_paths.py`
to remap.

## Files

| File | Rows | Notes |
|---|---|---|
| `train.jsonl` | ~280K | Used by `30-train_omnigene5_stage1.py` (vision warmup) and the mixed-stage pools downstream |
| `val.jsonl` | ~5K | Held-out validation slice |
| `meta.json` | — | Per-modality counts and split sizes |
| `B_chebi20/train.json` | ~26K | Vis-CheBI20 train split (project copy of [PharMolix/Vis-CheBI20](https://huggingface.co/datasets/PharMolix/Vis-CheBI20)). Five OCSU subtasks: `struct_recog`, `struct_cap`, `general_desp`, `trans_iupac`, `trans_smiles`. |
| `B_chebi20/test.json` | ~3K | Vis-CheBI20 test split, same five subtasks. Used by `60-eval_stage2.py`, `91-eval_stage3v2.py`, `92-eval_stage3v3.py`, `95-qualitative_demo_stage3v3.py`. |

## Citation

```bibtex
@article{wang2026omnigene4,
  author    = {Wang, Liang},
  title     = {{OmniGene-4}: A Unified Bio-Language MoE Model with Router-Level
               Interpretability and Modality-Invariant Transfer},
  year      = {2026},
  journal   = {bioRxiv},
  doi       = {10.64898/2026.05.12.724542}
}
```
README
  upload_file_with_retry "$REPO" /tmp/mm_readme.md "README.md"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/omnigene5/data/unified/train.jsonl "train.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/omnigene5/data/unified/val.jsonl "val.jsonl"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/omnigene5/data/unified/meta.json "meta.json"
  # ChEBI20 splits used for vision warmup + final eval (Vis-CheBI20 from PharMolix, project copy)
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20/train.json "B_chebi20/train.json"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20/test.json  "B_chebi20/test.json"
  echo "  $REPO done"
}

# ============================================================
# Repo 3: omnigene4-vis-chebi20 (project-side copy)
# ============================================================
upload_chebi() {
  REPO="dnagpt/omnigene4-vis-chebi20"
  echo "=== $REPO ==="
  hf_create_dataset "$REPO"

  cat > /tmp/chebi_readme.md <<'README'
---
license: cc-by-4.0
size_categories:
- 10K<n<100K
task_categories:
- image-text-to-text
tags:
- chemistry
- molecule
- multimodal
---

# Vis-CheBI20 (project copy used in OmniGene-4-MM)

A project-side copy of the [PharMolix/Vis-CheBI20](https://huggingface.co/datasets/PharMolix/Vis-CheBI20)
dataset (Fan et al. 2025, OCSU paper) used for OmniGene-4-MM Stage 1
vision warmup.

If you are reproducing OmniGene-4-MM, prefer the upstream PharMolix
release. This copy is provided to lock in the exact JSON splits used
in our paper.

## Files

| File | Description |
|---|---|
| `train.json` | Training set, five OCSU subtasks |
| `test.json` | Test set, five OCSU subtasks |

## OCSU subtasks

- `struct_recog` (highlighted functional-group recognition)
- `struct_cap` (functional-group caption)
- `general_desp` (free-form chemist-readable description)
- `trans_iupac` (image → IUPAC name)
- `trans_smiles` (image → SMILES, OCSR)

## Citation

```bibtex
@article{fan2025ocsu,
  title  = {OCSU: Optical Chemical Structure Understanding for Molecule-centric
            Scientific Discovery},
  author = {Fan, Siqi and Xie, Yuguang and Cai, Bowen and others},
  year   = {2025},
  journal= {arXiv:2501.15415}
}
@article{wang2026omnigene4,
  author = {Wang, Liang},
  title  = {{OmniGene-4}: A Unified Bio-Language MoE Model with Router-Level
            Interpretability and Modality-Invariant Transfer},
  year   = {2026},
  journal= {bioRxiv},
  doi    = {10.64898/2026.05.12.724542}
}
```
README
  upload_file_with_retry "$REPO" /tmp/chebi_readme.md "README.md"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20/train.json "train.json"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/omnigene5/data/B_chebi20/test.json "test.json"
  echo "  $REPO done"
}

# ============================================================
# Repo 4: omnigene4-cpt-corpus
# ============================================================
upload_cpt() {
  REPO="dnagpt/omnigene4-cpt-corpus"
  echo "=== $REPO ==="
  hf_create_dataset "$REPO"

  cat > /tmp/cpt_readme.md <<'README'
---
license: cc-by-4.0
language:
- en
- zh
size_categories:
- n>1M
task_categories:
- text-generation
tags:
- biology
- protein
- DNA
- 3Di
- DSSP
- bioinformatics
- continued-pretraining
---

# OmniGene-4 CPT corpus

Continued-pre-training (CPT) corpus for OmniGene-4 (see
https://github.com/maris205/omnigene4 ). Total ~96 GB across DNA,
protein, structure, and English-text replay splits.

## Files

| File | Size | Source / Description |
|---|---|---|
| `dna_32g.txt` | 31 GB | DNA sequences sampled from public genomes |
| `protein_uni_16.txt` | 16 GB | UniRef-derived protein sequences |
| `protein_lucaone_15g.txt` | 15 GB | Protein sequences from the LucaOne pretraining pool |
| `openwebtext.txt` | 37 GB | English text replay split (sampled from `Skylion007/openwebtext`); kept here so the entire CPT mixture is reproducible from a single repo |
| `pdb_aa.fasta` | 100 MB | PDB amino-acid sequences (paired with the 3Di split below) |
| `pdb_3di.fasta` | 100 MB | PDB Foldseek-3Di sequences (paired with `pdb_aa.fasta`) |
| `ss.txt` | 263 MB | DSSP secondary-structure labels |

## Mixing ratio used in training

CPT runs on a 1 : 1 : 1 split of DNA : protein : OpenWebText, with the
3Di + DSSP files used for the dual-head per-residue classification
objective in v5. See the OmniGene-4 paper (Methods §4.3) for the exact
recipe.

## Citation

```bibtex
@article{wang2026omnigene4,
  author    = {Wang, Liang},
  title     = {{OmniGene-4}: A Unified Bio-Language MoE Model with Router-Level
               Interpretability and Modality-Invariant Transfer},
  year      = {2026},
  journal   = {bioRxiv},
  doi       = {10.64898/2026.05.12.724542}
}
```
README
  upload_file_with_retry "$REPO" /tmp/cpt_readme.md "README.md"
  # smaller first
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/data/pdb_aa.fasta              "pdb_aa.fasta"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/data/pdb_3di.fasta             "pdb_3di.fasta"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/data/ss.txt                    "ss.txt"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/data/protein_lucaone_15g.txt   "protein_lucaone_15g.txt"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/data/protein_uni_16.txt        "protein_uni_16.txt"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/data/dna_32g.txt               "dna_32g.txt"
  upload_file_with_retry "$REPO" /root/autodl-tmp/dnagpt/data/openwebtext.txt           "openwebtext.txt"
  echo "  $REPO done"
}

# ---------- dispatch ----------
case "$WHICH" in
  small)
    upload_sft
    upload_mm
    ;;
  cpt)
    upload_cpt
    ;;
  all)
    upload_sft
    upload_mm
    upload_cpt
    ;;
  *)
    echo "Usage: $0 [small|cpt|all]"
    exit 1
    ;;
esac

echo "ALL UPLOADS DONE: $WHICH"
