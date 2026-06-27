"""Instruction-template bank for BioPAWS-2 converters.

Each task_id maps to a SMALL SET of paraphrased instruction templates. Converters cycle
through them deterministically (by sample index, no RNG — scripts run in a no-Date/no-random
context). Template diversity prevents SFT models from overfitting a single prompt surface
and makes zero-shot eval fairer across instruction phrasings.

Templates use Python str.format with named fields. `{seq1}`, `{seq2}`, `{seq}`, `{smiles}`
etc. are filled by the converter.
"""
from __future__ import annotations

# ---- F1 pairwise homology ----------------------------------------------------

PROTEIN_HOMOLOGY = [
    ("Determine whether the following two protein sequences are homologous "
     "(evolutionarily related, sharing a common ancestor and fold).\n"
     "Sequence 1: {seq1}\nSequence 2: {seq2}\n"
     "Answer with 'Yes' (homologous) or 'No' (non-homologous)."),
    ("Are these two proteins homologs? Homologous proteins descend from a common "
     "ancestor and typically share structural fold despite sequence divergence.\n"
     "Protein A: {seq1}\nProtein B: {seq2}\nReply Yes or No."),
    ("Two amino-acid sequences are given. Judge structural/evolutionary homology.\n"
     ">seq1\n{seq1}\n>seq2\n{seq2}\nIs there homology? (Yes/No)"),
]

PROTEIN_HOMOLOGY_REMOTE = [
    ("These two proteins may share less than 25% sequence identity (the 'twilight zone'). "
     "Decide whether they are nonetheless remote homologs sharing a common fold.\n"
     "Sequence 1: {seq1}\nSequence 2: {seq2}\nAnswer Yes or No."),
    ("Remote-homology judgment (low sequence identity, alignment-free). Do these two "
     "sequences adopt the same fold / belong to the same superfamily?\n"
     "A: {seq1}\nB: {seq2}\nYes or No?"),
]

DNA_HOMOLOGY = [
    ("Determine whether the following two DNA sequences are homologous.\n"
     "Sequence 1: {seq1}\nSequence 2: {seq2}\nAnswer Yes or No."),
    ("Are these two nucleotide sequences evolutionarily related (homologous)?\n"
     ">dna1\n{seq1}\n>dna2\n{seq2}\nReply Yes or No."),
]

CENTRAL_DOGMA = [
    ("Does the coding DNA sequence (CDS) below translate into the given protein, "
     "consistent with the Central Dogma?\nCDS: {seq1}\nProtein: {seq2}\nAnswer Yes or No."),
    ("Verify the DNA→protein correspondence. Is this protein the translation product of "
     "this coding sequence?\nDNA: {seq1}\nProtein: {seq2}\nYes or No?"),
]

# Map answers
YESNO = {1: "Yes", 0: "No"}

# Optional explanatory tails appended to the assistant answer for CoT-style SFT.
HOMOLOGY_EXPLAIN = {
    1: " The sequences share conserved residue patterns consistent with a common fold.",
    0: " The sequences lack the conserved structural signature of a shared fold.",
}


def template_for(task_id: str):
    return {
        "protein_homology_std": PROTEIN_HOMOLOGY,
        "protein_homology_remote": PROTEIN_HOMOLOGY_REMOTE,
        "dna_homology": DNA_HOMOLOGY,
        "central_dogma": CENTRAL_DOGMA,
    }[task_id]
