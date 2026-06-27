"""BioPAWS-2 scoring: metric implementations + per-task / per-family aggregation.

Given gold records (jsonl with answer_short / meta.value / metric) and a model's
predictions (id -> raw text), compute the appropriate score per task and aggregate.

Pure-stdlib where possible; scipy/sklearn used only if available (graceful fallback).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# Semantic answer aliases: a model may phrase a binary answer many ways. These map the
# free-text surface form to a canonical choice so scoring is fair across model vocabularies
# (e.g. OmniGene says "Homologous", Qwen says "Yes", another says "True"). ORDER MATTERS:
# negatives are checked before positives because "non-homologous" contains "homologous".
_ALIAS_NEG = ["nonhomologous", "nothomologous", "nonhomolog", "different", "false",
              "negative", "no", "incorrect", "unrelated", "donotmatch", "notrelated"]
_ALIAS_POS = ["homologous", "homolog", "same", "true", "positive", "yes", "correct",
              "related", "match", "paraphrase"]


def _binary_alias(pred_norm: str, choice_norm: str) -> bool:
    """Does pred (normalized) express the given Yes/No-style choice via aliases?"""
    if choice_norm in ("no", "nonhomologous", "false", "negative", "different"):
        return any(a in pred_norm for a in _ALIAS_NEG)
    if choice_norm in ("yes", "homologous", "true", "positive", "same"):
        # ensure it's not actually a negative phrasing
        if any(a in pred_norm for a in _ALIAS_NEG):
            return False
        return any(a in pred_norm for a in _ALIAS_POS)
    # variant-effect vocabulary (ProteinGym F4): deleterious vs benign
    if choice_norm == "deleterious":
        return any(a in pred_norm for a in
                   ("deleterious", "pathogenic", "damaging", "nonfunctional",
                    "lossoffunction", "harmful", "destabiliz")) \
            and "nonpathogenic" not in pred_norm and "notpathogenic" not in pred_norm
    if choice_norm == "benign":
        if any(a in pred_norm for a in ("deleterious", "pathogenic", "damaging",
                                        "nonfunctional", "harmful")) \
           and "nonpathogenic" not in pred_norm:
            return False
        return any(a in pred_norm for a in
                   ("benign", "nonpathogenic", "tolerated", "neutral", "functional",
                    "noeffect", "wildtype", "harmless"))
    return False


def _extract_choice(pred: str, choices: list[str]) -> str | None:
    """Find which choice the prediction expresses (robust to verbose answers)."""
    p = _norm(pred)
    # 1) exact normalized containment, longest choice first ('No' before 'Nonsense')
    for c in sorted(choices, key=len, reverse=True):
        if _norm(c) and _norm(c) in p:
            return c
    # 2) binary semantic aliases (Yes/No-style 2-choice tasks only)
    if len(choices) == 2:
        # check negative/deleterious choice first (substring safety)
        ordered = sorted(choices, key=lambda c: 0 if _norm(c) in
                         ("no", "nonhomologous", "false", "negative", "different",
                          "deleterious") else 1)
        for c in ordered:
            if _binary_alias(p, _norm(c)):
                return c
    return None


def _extract_number(pred: str) -> float | None:
    m = re.search(r"-?\d+\.?\d*", str(pred))
    return float(m.group()) if m else None


# ---- metrics -----------------------------------------------------------------

def acc_score(golds, preds, choices_map):
    correct = total = 0
    for g in golds:
        pid = g["id"]
        if pid not in preds:
            continue
        total += 1
        ch = g.get("choices") or choices_map.get(g["task_id"])
        pred_c = _extract_choice(preds[pid], ch) if ch else None
        if pred_c is not None and _norm(pred_c) == _norm(g["answer_short"]):
            correct += 1
        elif ch is None and _norm(preds[pid]).find(_norm(g["answer_short"])) >= 0:
            correct += 1
    return correct / total if total else 0.0, total


def mcc_score(golds, preds, choices_map):
    # binary MCC over the dominant 2-class set (genomics convention)
    tp = tn = fp = fn = 0
    for g in golds:
        pid = g["id"]
        if pid not in preds:
            continue
        ch = g.get("choices") or choices_map.get(g["task_id"]) or ["Yes", "No"]
        gold = _norm(g["answer_short"])
        pred_c = _extract_choice(preds[pid], ch)
        pred = _norm(pred_c) if pred_c else ""
        pos = _norm(ch[0])
        if gold == pos and pred == pos:
            tp += 1
        elif gold != pos and pred != pos and pred:
            tn += 1
        elif gold != pos and pred == pos:
            fp += 1
        elif gold == pos and pred != pos:
            fn += 1
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return mcc, tp + tn + fp + fn


def spearman_score(golds, preds, choices_map):
    """Bucketized regression: rank model's ordinal/number vs meta.value."""
    xs, ys = [], []
    ord_rank = {"low": 0, "medium": 1, "high": 2}
    for g in golds:
        pid = g["id"]
        if pid not in preds or "value" not in (g.get("meta") or {}):
            continue
        gold_val = g["meta"]["value"]
        pred = preds[pid]
        # try ordinal word first, else a raw number
        pc = _extract_choice(pred, ["low", "medium", "high"])
        if pc is not None:
            pv = ord_rank[pc.lower()]
        else:
            pv = _extract_number(pred)
            if pv is None:
                continue
        xs.append(gold_val)
        ys.append(pv)
    if len(xs) < 3:
        return 0.0, len(xs)
    try:
        from scipy.stats import spearmanr
        rho = spearmanr(xs, ys).correlation
        return (rho if rho == rho else 0.0), len(xs)
    except Exception:
        # manual Spearman
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            for pos, idx in enumerate(order):
                r[idx] = pos
            return r
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
        return 1 - 6 * d2 / (n * (n * n - 1)), n


def text_overlap_score(golds, preds, choices_map):
    """ROUGE-L-ish unigram F1 fallback for free-form (rouge_l/bleu)."""
    def f1(a, b):
        ta, tb = _norm(a), _norm(b)
        sa = set(re.findall(r"[a-z0-9]+", a.lower()))
        sb = set(re.findall(r"[a-z0-9]+", b.lower()))
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        p = inter / len(sb)
        r = inter / len(sa)
        return 2 * p * r / (p + r) if (p + r) else 0.0
    scores = []
    for g in golds:
        pid = g["id"]
        if pid not in preds:
            continue
        scores.append(f1(g["answer_short"], preds[pid]))
    return (sum(scores) / len(scores) if scores else 0.0), len(scores)


METRIC_FN = {
    "accuracy": acc_score,
    "f1": acc_score,          # macro-F1 approximated by accuracy in MC setting
    "mcc": mcc_score,
    "spearman": spearman_score,
    "exact_match": acc_score,
    "rouge_l": text_overlap_score,
    "bleu": text_overlap_score,
}


def score_task(gold_path: str, preds: dict[str, str]) -> dict:
    """Score one task jsonl (test split) against preds dict {id: text}."""
    golds = []
    choices_map = {}
    metric = "accuracy"
    with open(gold_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("split") != "test":
                continue
            golds.append(r)
            metric = r.get("metric", metric)
            if r.get("choices"):
                choices_map[r["task_id"]] = r["choices"]
    fn = METRIC_FN.get(metric, acc_score)
    val, n = fn(golds, preds, choices_map)
    return {"metric": metric, "score": round(val, 4), "n": n}


def aggregate(per_task: dict[str, dict]) -> dict:
    """Family-level and overall macro averages."""
    fam_scores = defaultdict(list)
    for task_id, res in per_task.items():
        fam = res.get("family", "?")
        fam_scores[fam].append(res["score"])
    fam_avg = {f: round(sum(v) / len(v), 4) for f, v in fam_scores.items()}
    overall = round(sum(fam_avg.values()) / len(fam_avg), 4) if fam_avg else 0.0
    return {"per_family": fam_avg, "overall_macro": overall}


if __name__ == "__main__":
    import sys
    # quick smoke: score a gold file against a perfect oracle
    gold = sys.argv[1]
    preds = {}
    with open(gold, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line.strip())
            if r.get("split") == "test":
                preds[r["id"]] = r["answer_short"]
    print("oracle:", score_task(gold, preds))
