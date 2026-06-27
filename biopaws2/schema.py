"""BioPAWS-2 canonical sample schema + validator.

Design goals (per project spec):
  1. FORMAT IS THE CONTRIBUTION. Every sample is an instruction-tuning chat record,
     NOT a (sequence, label) row. This is what separates BioPAWS-2 from TAPE / PEER /
     ProteinGym / GUE (PLM + classification head) and from plain text-QA benchmarks.
  2. COVERS TRADITIONAL TEST SETS. Any classification / regression / retrieval label can
     be expressed here: classification -> choices, regression -> ordinal bucket
     (high/med/low) with the raw value preserved in `meta.value` for Spearman scoring.
  3. EXTENSIBLE. New task families / subtasks plug in by emitting records that pass
     `validate()`. No schema change needed to add a task.
  4. DUAL-MODE READY. The same record drives zero-shot eval (read `messages`),
     SFT (train on `messages`), and exact-match/numeric scoring (read `answer_short`).
  5. MULTIMODAL. `images` + `<image>` placeholders make vision QA first-class.

Wire-compatible with the OmniGene-4-MM unified corpus (omnigene5/scripts/01-build_unified_jsonl.py),
so existing training / eval harnesses consume BioPAWS-2 jsonl unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

# ---- controlled vocabularies -------------------------------------------------

# 9 task families. Extend by appending; validator only checks membership.
TASK_FAMILIES = {
    "F1_pairwise",      # homology / Central Dogma pairwise alignment (BioPAWS-1 core)
    "F2_functional",    # protein single-sequence functional (EC/GO/loc/solubility/Tm)
    "F3_dna",           # DNA / genomic classification (promoter/enhancer/splice/TFBS...)
    "F4_variant",       # variant / mutation effect (ProteinGym/ClinVar/ddG)
    "F5_structure",     # structure-as-text (SCOPe fold / SS / 3Di)
    "F6_crossmodal",    # DNA+protein+molecule mixed (DTI/PPI/CDS-protein/SMILES-prop)
    "F7_cot",           # biological chain-of-thought reasoning
    "F8_biomed_qa",     # biomedical sequence-grounded QA (BixBench-style)
    "F9_multimodal",    # image+text QA (Vis-CheBI20 / HPA / chart / image->SMILES)
}

MODALITIES = {"dna", "protein", "structure", "smiles", "vision", "text"}

# metric -> how `answer_short` is scored against gold
METRICS = {
    "accuracy",     # exact categorical match
    "f1",           # macro-F1 over classes
    "mcc",          # Matthews corr (binary genomics convention)
    "spearman",     # ranking corr on meta.value (bucketized regression)
    "exact_match",  # string EM (e.g. EC "3.4.21")
    "rouge_l",      # free-form description
    "bleu",         # captioning / translation
}

ORDINAL_BUCKETS = ["low", "medium", "high"]  # canonical regression->QA buckets

REQUIRED_FIELDS = (
    "id", "task_family", "task_id", "modality", "images",
    "messages", "answer_short", "metric", "split", "license", "source",
)
SPLITS = {"train", "val", "test"}


@dataclass
class Sample:
    """One BioPAWS-2 instruction-tuning QA record."""
    id: str
    task_family: str
    task_id: str
    modality: list[str]
    messages: list[dict[str, str]]      # [{role: user|assistant, content: str}, ...]
    answer_short: str                    # canonical gold for scoring
    metric: str
    split: str
    license: str
    source: str
    images: list[str] = field(default_factory=list)
    choices: list[str] | None = None     # present for multiple-choice
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. {"value": 0.83} for regression

    def to_json(self) -> str:
        d = asdict(self)
        if d["choices"] is None:
            d.pop("choices")
        if not d["meta"]:
            d.pop("meta")
        return json.dumps(d, ensure_ascii=False)


def validate(rec: dict[str, Any]) -> list[str]:
    """Return a list of problems; empty list == valid. Cheap, no I/O."""
    errs: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in rec:
            errs.append(f"missing field: {f}")
    if errs:
        return errs

    if rec["task_family"] not in TASK_FAMILIES:
        errs.append(f"unknown task_family: {rec['task_family']}")
    if rec["metric"] not in METRICS:
        errs.append(f"unknown metric: {rec['metric']}")
    if rec["split"] not in SPLITS:
        errs.append(f"unknown split: {rec['split']}")

    mod = rec["modality"]
    if not isinstance(mod, list) or not mod:
        errs.append("modality must be a non-empty list")
    else:
        for m in mod:
            if m not in MODALITIES:
                errs.append(f"unknown modality: {m}")

    msgs = rec["messages"]
    if not isinstance(msgs, list) or len(msgs) < 2:
        errs.append("messages must have >=2 turns (user + assistant)")
    else:
        roles = [m.get("role") for m in msgs]
        if roles[0] != "user":
            errs.append("first message must be role=user")
        if roles[-1] != "assistant":
            errs.append("last message must be role=assistant")
        for m in msgs:
            if m.get("role") not in {"user", "assistant", "system"}:
                errs.append(f"bad role: {m.get('role')}")
            if not str(m.get("content", "")).strip():
                errs.append("empty message content")

    # vision sanity: <image> placeholders must match number of images
    n_img = len(rec.get("images") or [])
    n_ph = sum(str(m.get("content", "")).count("<image>") for m in msgs)
    if "vision" in (mod if isinstance(mod, list) else []):
        if n_img == 0:
            errs.append("vision modality but no images")
    if n_img and n_ph and n_ph != n_img:
        errs.append(f"<image> placeholder count {n_ph} != images {n_img}")

    if not str(rec["answer_short"]).strip():
        errs.append("answer_short empty")

    # regression->QA: spearman metric needs a numeric back-channel
    if rec["metric"] == "spearman":
        if "value" not in (rec.get("meta") or {}):
            errs.append("metric=spearman requires meta.value (raw numeric)")

    # multiple-choice consistency
    ch = rec.get("choices")
    if ch is not None:
        if not isinstance(ch, list) or len(ch) < 2:
            errs.append("choices must be a list of >=2")
        elif rec["metric"] in {"accuracy", "f1", "mcc"} and rec["answer_short"] not in ch:
            errs.append("answer_short not among choices")

    return errs


def validate_file(path: str, max_report: int = 20) -> tuple[int, int, list[str]]:
    """Validate a jsonl file. Returns (n_total, n_bad, sample_errors)."""
    n_total = n_bad = 0
    reports: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                n_bad += 1
                if len(reports) < max_report:
                    reports.append(f"line {i}: JSON error: {e}")
                continue
            errs = validate(rec)
            if errs:
                n_bad += 1
                if len(reports) < max_report:
                    reports.append(f"line {i} ({rec.get('id','?')}): {'; '.join(errs)}")
    return n_total, n_bad, reports


# ---- helpers for converters --------------------------------------------------

def bucketize(value: float, edges: tuple[float, float]) -> str:
    """Map a scalar to low/medium/high given two threshold edges (lo, hi)."""
    lo, hi = edges
    if value < lo:
        return "low"
    if value < hi:
        return "high" if lo == hi else "medium"
    return "high"


def make_sample(**kw) -> Sample:
    """Construct + eagerly validate a Sample (raises on invalid)."""
    s = Sample(**kw)
    errs = validate(json.loads(s.to_json()))
    if errs:
        raise ValueError(f"invalid sample {kw.get('id')}: {errs}")
    return s


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            n, bad, reps = validate_file(p)
            print(f"{p}: {n} samples, {bad} invalid")
            for r in reps:
                print("  ", r)
    else:
        # self-test
        demo = make_sample(
            id="f2_ec_number:00012",
            task_family="F2_functional",
            task_id="ec_number",
            modality=["protein", "text"],
            messages=[
                {"role": "user", "content": "Predict the top-level EC class.\nSequence: MKTAYIAK..."},
                {"role": "assistant", "content": "EC 3 (Hydrolase)."},
            ],
            answer_short="3",
            choices=["1", "2", "3", "4", "5", "6", "7"],
            metric="accuracy",
            split="train",
            license="CC-BY-4.0",
            source="UniProt:P00761",
        )
        print("self-test OK:", demo.to_json()[:120], "...")
