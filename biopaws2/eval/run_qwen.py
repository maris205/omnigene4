"""Qwen-3 (and other OpenAI-compatible) remote-model runner for BioPAWS-2 zero-shot.

Concurrent chat-completions over a task's test split. Scores via eval/score.py. This is the
massive-LLM reference tier (BioPAWS-1 showed Qwen-3 ~100% std / 65-75% remote). No GPU used.

Usage:
  python eval/run_qwen.py --task-file data/protein_homology_std.jsonl --limit 500
  python eval/run_qwen.py --task-file data/lg_fold_class.jsonl --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402
from eval._apikeys import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL  # noqa: E402


def load_test(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("split") == "test":
                    rows.append(r)
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--model", default=QWEN_MODEL)
    ap.add_argument("--limit", type=int, default=500,
                    help="cap test rows (API cost); default 500 like BioPAWS-1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    task = os.path.basename(a.task_file).replace(".jsonl", "")
    rows = load_test(a.task_file, a.limit)
    tag = a.tag or a.model
    print(f"[qwen] {task}: {len(rows)} test rows | model={a.model} workers={a.workers}")

    def ask(r):
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        # text-only: drop image placeholders (Qwen text endpoint)
        content = "\n".join(m["content"].replace("<image>", "") for m in msgs)
        try:
            resp = client.chat.completions.create(
                model=a.model,
                messages=[{"role": "user", "content": content}],
                temperature=0, max_tokens=a.max_tokens)
            return r["id"], resp.choices[0].message.content.strip()
        except Exception as e:
            return r["id"], f"[error] {e}"

    preds = {}
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(ask, r) for r in rows]
        for f in as_completed(futs):
            pid, text = f.result()
            preds[pid] = text
            done += 1
            if done % 50 == 0:
                print(f"  [qwen] {done}/{len(rows)}", flush=True)

    n_err = sum(1 for v in preds.values() if v.startswith("[error]"))
    if n_err:
        print(f"  [qwen] WARNING {n_err} API errors")
    res = score_task(a.task_file, preds)
    res.update({"mode": "zeroshot", "model": tag, "task": task, "n_api_errors": n_err})
    os.makedirs(a.out_dir, exist_ok=True)
    safe = tag.replace("/", "_")
    out = os.path.join(a.out_dir, f"{safe}__{task}.zeroshot.json")
    json.dump({"result": res, "predictions": preds},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[qwen] {res}  -> {out}")


if __name__ == "__main__":
    main()
