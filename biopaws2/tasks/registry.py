"""BioPAWS-2 task registry.

Single source of truth for every subtask: its family, data source, license, default
metric, and the converter module that produces its QA jsonl. `scripts/build_all.py` and
the leaderboard read this. Adding a task = appending one TaskSpec (extensibility contract).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    family: str          # one of schema.TASK_FAMILIES
    task_id: str         # unique subtask id
    converter: str       # dotted module path under tasks/ implementing build(out_dir)
    source: str          # human-readable provenance
    license: str
    metric: str          # default scoring metric (schema.METRICS)
    modality: tuple[str, ...]
    status: str = "planned"  # planned | ready | done
    note: str = ""


REGISTRY: list[TaskSpec] = [
    # ---- F1 Pairwise (BioPAWS-1 core, local data ready) ---------------------
    TaskSpec("F1_pairwise", "protein_homology_std", "f1_pairwise.protein_homology",
             "dnagpt/biopaws:protein_pair_short", "CC-BY-4.0", "accuracy",
             ("protein", "text"), status="ready",
             note="standard homology, 20K balanced pairs"),
    TaskSpec("F1_pairwise", "protein_homology_remote", "f1_pairwise.protein_homology",
             "dnagpt/biopaws:protein_pair_remote", "CC-BY-4.0", "accuracy",
             ("protein", "text"), status="ready",
             note="remote homology <25% ID, 359K balanced pairs (subsample)"),
    TaskSpec("F1_pairwise", "dna_homology", "f1_pairwise.dna_homology",
             "biopaws DNA pair", "CC-BY-4.0", "accuracy", ("dna", "text"),
             status="planned"),
    TaskSpec("F1_pairwise", "central_dogma", "f1_pairwise.central_dogma",
             "biopaws CDS-protein", "CC-BY-4.0", "accuracy", ("dna", "protein", "text"),
             status="planned"),

    # ---- F2 Functional (needs UniProt / DeepLoc / Meltome downloads) ---------
    TaskSpec("F2_functional", "ec_number", "f2_functional.ec_number",
             "UniProt/SwissProt EC", "CC-BY-4.0", "accuracy", ("protein", "text")),
    TaskSpec("F2_functional", "subcellular_loc", "f2_functional.subcellular_loc",
             "DeepLoc 2.0", "CC-BY-4.0", "accuracy", ("protein", "text")),
    TaskSpec("F2_functional", "solubility", "f2_functional.solubility",
             "DeepSol", "CC-BY-4.0", "accuracy", ("protein", "text")),
    TaskSpec("F2_functional", "thermostability", "f2_functional.thermostability",
             "Meltome Atlas (FLIP)", "CC-BY-4.0", "spearman", ("protein", "text")),

    # ---- F3 DNA (GUE / DeepSEA / Genomic Benchmarks) ------------------------
    TaskSpec("F3_dna", "promoter", "f3_dna.promoter", "GUE / Genomic Benchmarks",
             "CC-BY-4.0", "mcc", ("dna", "text")),
    TaskSpec("F3_dna", "splice_site", "f3_dna.splice_site", "SpliceAI / GUE",
             "CC-BY-4.0", "accuracy", ("dna", "text")),
    TaskSpec("F3_dna", "tf_binding", "f3_dna.tf_binding", "GUE 690 ENCODE",
             "CC-BY-4.0", "mcc", ("dna", "text")),
    TaskSpec("F3_dna", "mrl_5utr", "f3_dna.mrl_5utr", "Optimus 5-Prime MPRA",
             "CC-BY-4.0", "spearman", ("dna", "text")),

    # ---- F4 Variant ----------------------------------------------------------
    TaskSpec("F4_variant", "dms_fitness", "f4_variant.dms_fitness", "ProteinGym v1.1",
             "MIT", "spearman", ("protein", "text")),
    TaskSpec("F4_variant", "clinvar_path", "f4_variant.clinvar_path", "ClinVar",
             "public-domain", "accuracy", ("protein", "text")),
    TaskSpec("F4_variant", "ddg_stability", "f4_variant.ddg_stability", "MegaScale",
             "CC-BY-4.0", "spearman", ("protein", "text")),

    # ---- F5 Structure --------------------------------------------------------
    TaskSpec("F5_structure", "scope_fold", "f5_structure.scope_fold", "SCOPe 2.08",
             "free-academic", "accuracy", ("protein", "structure", "text")),
    TaskSpec("F5_structure", "ss3", "f5_structure.secondary_structure", "TAPE SS-Q3",
             "CC-BY-4.0", "accuracy", ("protein", "text")),
    TaskSpec("F5_structure", "ss8", "f5_structure.secondary_structure", "DSSP",
             "CC-BY-4.0", "accuracy", ("protein", "text")),

    # ---- F6 Crossmodal -------------------------------------------------------
    TaskSpec("F6_crossmodal", "dti_binding", "f6_crossmodal.dti", "DAVIS / KIBA",
             "CC-BY-4.0", "spearman", ("protein", "smiles", "text")),
    TaskSpec("F6_crossmodal", "ppi", "f6_crossmodal.ppi", "STRING / PEER",
             "CC-BY-4.0", "accuracy", ("protein", "text")),
    TaskSpec("F6_crossmodal", "smiles_property", "f6_crossmodal.smiles_property",
             "MoleculeNet ESOL/Lipo/FreeSolv", "MIT", "spearman", ("smiles", "text")),
    TaskSpec("F6_crossmodal", "prot2text", "f6_crossmodal.prot2text",
             "UniProt comments / Prot2Text", "CC-BY-4.0", "rouge_l", ("protein", "text")),

    # ---- F7 CoT --------------------------------------------------------------
    TaskSpec("F7_cot", "mental_folding", "f7_cot.mental_folding",
             "BioPAWS-1 Qwen-3 traces", "CC-BY-4.0", "rouge_l", ("protein", "text")),

    # ---- F8 Biomed QA --------------------------------------------------------
    TaskSpec("F8_biomed_qa", "seq_grounded_tf", "f8_biomed_qa.seq_grounded",
             "BixBench-style", "CC-BY-4.0", "accuracy", ("protein", "text")),

    # ---- F9 Multimodal (reuse OmniGene-4-MM corpus) -------------------------
    TaskSpec("F9_multimodal", "mol_recog", "f9_multimodal.vis_chebi20",
             "Vis-CheBI20 struct_recog", "CC-BY-4.0", "accuracy", ("vision", "text"),
             status="ready", note="reuse omnigene5 unified corpus"),
    TaskSpec("F9_multimodal", "mol_caption", "f9_multimodal.vis_chebi20",
             "Vis-CheBI20 struct_cap", "CC-BY-4.0", "rouge_l", ("vision", "text"),
             status="ready"),
    TaskSpec("F9_multimodal", "hpa_localization", "f9_multimodal.hpa",
             "Human Protein Atlas", "CC-BY-SA-3.0", "accuracy", ("vision", "text")),
]


def by_family() -> dict[str, list[TaskSpec]]:
    out: dict[str, list[TaskSpec]] = {}
    for t in REGISTRY:
        out.setdefault(t.family, []).append(t)
    return out


def summary() -> str:
    fams = by_family()
    lines = [f"BioPAWS-2 registry: {len(REGISTRY)} subtasks across {len(fams)} families"]
    for fam in sorted(fams):
        ts = fams[fam]
        ready = sum(1 for t in ts if t.status == "ready")
        lines.append(f"  {fam}: {len(ts)} subtasks ({ready} ready)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
