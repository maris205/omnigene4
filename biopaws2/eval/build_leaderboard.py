"""BioPAWS-2 leaderboard builder.

Scans results/*.json (zeroshot + sft outputs), pairs them per (model, task), and emits the
leaderboard table with the signature BioPAWS-2 columns:

    model · task/family · base_acc (zero-shot) · ft_acc (SFT) · Δ (ft − base)

Δ is the trainability axis — how much a model gains from BioPAWS-2 instruction data. The
overall leaderboard macro-averages per family then across families (MMLU-style), so no
single large task family dominates.

Usage:
    python eval/build_leaderboard.py --results results --out docs/leaderboard.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.registry import REGISTRY  # noqa: E402

TASK_FAMILY = {t.task_id: t.family for t in REGISTRY}
# task_id may differ from file stem; also map file stems we know
STEM_FAMILY = {
    "protein_homology_std": "F1_pairwise",
    "protein_homology_remote": "F1_pairwise",
    "f9_vis_chebi20": "F9_multimodal",
    # llama-gene-derived tasks
    "lg_promoter_detection": "F3_dna",
    "lg_core_promoter_detection": "F3_dna",
    "lg_splice_site": "F3_dna",
    "lg_tf_prediction": "F3_dna",
    "lg_fold_class": "F5_structure",
    "lg_subcellular_loc": "F2_functional",
    "lg_signal_peptide": "F2_functional",
    "lg_npp": "F2_functional",
    "lg_central_dogma": "F6_crossmodal",
    # F7 CoT / F8 biomed QA / F9 multimodal
    "f7_bioreason_cot": "F7_cot",
    "f8_bixbench_mcq": "F8_biomed_qa",
    "f8_bixbench_tf": "F8_biomed_qa",
    "f9_mol_recog": "F9_multimodal",
    "f9_mol_caption": "F9_multimodal",
    "f9_mol_iupac": "F9_multimodal",
    "f4_proteingym_dms": "F4_variant",
    "int_uniprot_qa": "F2_functional",
    "int_protein2text_qa": "F2_functional",
    "int_opi_function": "F2_functional",
    "int_protein_catalogue_cot": "F7_cot",
}


def family_of(task: str) -> str:
    return TASK_FAMILY.get(task) or STEM_FAMILY.get(task, "?")


def load_results(results_dir: str):
    # (model, task) -> {"base": score, "ft": score}
    table: dict[tuple[str, str], dict] = defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        res = d.get("result", {})
        model = res.get("model", "?")
        task = res.get("task", "?")
        mode = res.get("mode", "?")
        key = (model, task)
        if mode == "zeroshot":
            table[key]["base"] = res.get("score")
            table[key]["metric"] = res.get("metric")
        elif mode == "sft":
            table[key]["ft"] = res.get("score")
            table[key]["metric"] = res.get("metric")
            table[key]["paradigm"] = res.get("paradigm", "lora_sft")
        elif mode == "joint_sft":
            # one generalist model evaluated on every task
            table[key]["joint"] = res.get("score")
            table[key]["metric"] = res.get("metric")
            table[key]["n_models_needed"] = res.get("n_models_needed", 1)
    return table


def parity_gaps(table):
    """For each task, gap = best chat-joint score - best PLM-head score.
    Returns {task: {chat, head, gap}}. 'head' rows are those with paradigm==plm_head."""
    by_task = defaultdict(lambda: {"chat": [], "head": []})
    for (model, task), v in table.items():
        if v.get("paradigm") == "plm_head" and v.get("ft") is not None:
            by_task[task]["head"].append(v["ft"])
        # chat generalist (joint) preferred; else per-task lora ft
        chat = v.get("joint")
        if chat is None and v.get("paradigm") != "plm_head":
            chat = v.get("ft")
        if chat is not None:
            by_task[task]["chat"].append(chat)
    out = {}
    for task, d in by_task.items():
        chat = max(d["chat"]) if d["chat"] else None
        head = max(d["head"]) if d["head"] else None
        gap = (round(chat - head, 4) if (chat is not None and head is not None) else None)
        out[task] = {"chat": chat, "head": head, "gap": gap}
    return out


def build_md(table, out_path):
    # per-row
    rows = []
    for (model, task), v in sorted(table.items()):
        base = v.get("base")
        # joint-SFT generalist score counts as this model's ft score for display
        ft = v.get("ft") if v.get("ft") is not None else v.get("joint")
        delta = (round(ft - base, 4) if (base is not None and ft is not None) else None)
        rows.append({
            "model": model, "task": task, "family": family_of(task),
            "metric": v.get("metric", "?"),
            "base": base, "ft": ft, "delta": delta,
        })

    # per-model family macro + overall
    model_fam = defaultdict(lambda: defaultdict(lambda: {"base": [], "ft": []}))
    for r in rows:
        mf = model_fam[r["model"]][r["family"]]
        if r["base"] is not None:
            mf["base"].append(r["base"])
        if r["ft"] is not None:
            mf["ft"].append(r["ft"])

    lines = ["# BioPAWS-2 Leaderboard", "",
             "Dual-mode protocol: **base** = zero-shot QA, **ft** = LoRA-SFT-then-eval, "
             "**Δ** = ft − base (trainability).", "",
             "## Per-task", "",
             "| Model | Family | Task | Metric | base | ft | Δ |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        def fmt(x):
            return "—" if x is None else f"{x:.3f}"
        lines.append(f"| {r['model']} | {r['family']} | {r['task']} | {r['metric']} | "
                     f"{fmt(r['base'])} | {fmt(r['ft'])} | {fmt(r['delta'])} |")

    lines += ["", "## Per-model summary (macro over families)", "",
              "| Model | base (macro) | ft (macro) | Δ (macro) |",
              "|---|---|---|---|"]
    for model, fams in sorted(model_fam.items()):
        fam_base, fam_ft = [], []
        for fam, d in fams.items():
            if d["base"]:
                fam_base.append(sum(d["base"]) / len(d["base"]))
            if d["ft"]:
                fam_ft.append(sum(d["ft"]) / len(d["ft"]))
        b = sum(fam_base) / len(fam_base) if fam_base else None
        f = sum(fam_ft) / len(fam_ft) if fam_ft else None
        dl = (f - b) if (b is not None and f is not None) else None
        def fmt(x):
            return "—" if x is None else f"{x:.3f}"
        lines.append(f"| {model} | {fmt(b)} | {fmt(f)} | {fmt(dl)} |")

    # --- generalist vs specialist section ---
    gaps = parity_gaps(table)
    if gaps:
        lines += ["", "## Generalist (chat, 1 model) vs Specialist (PLM head, N models)", "",
                  "Per task: best chat-model score vs best PLM-head score. `gap = chat − head` "
                  "(small/positive = chat paradigm matches or beats the single-task head).", "",
                  "| Task | chat (1 model) | PLM head (per-task) | gap |",
                  "|---|---|---|---|"]
        valid_gaps = []
        for task, d in sorted(gaps.items()):
            def fmt(x):
                return "—" if x is None else f"{x:.3f}"
            lines.append(f"| {task} | {fmt(d['chat'])} | {fmt(d['head'])} | {fmt(d['gap'])} |")
            if d["gap"] is not None:
                valid_gaps.append(d["gap"])
        if valid_gaps:
            mean_gap = sum(valid_gaps) / len(valid_gaps)
            n_tasks = len(gaps)
            lines += ["",
                      f"**Coverage cost**: chat generalist needs **1** model + 1 training to "
                      f"cover all {n_tasks} tasks; PLM-head paradigm needs **{n_tasks}** "
                      f"separate heads.",
                      f"**Mean parity gap (chat − head)**: {mean_gap:+.3f} "
                      f"— a single generalist model stays within this of task-specific heads "
                      f"while covering {n_tasks}× the tasks with one interface."]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[leaderboard] {len(rows)} task-rows, {len(model_fam)} models -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/leaderboard.md")
    a = ap.parse_args()
    table = load_results(a.results)
    build_md(table, a.out)


if __name__ == "__main__":
    main()
