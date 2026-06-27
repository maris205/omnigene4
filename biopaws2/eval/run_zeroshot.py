"""BioPAWS-2 Mode A: zero-shot QA evaluation.

Runs a model over the `test` split of one or more task jsonl files WITHOUT any fine-tuning,
collects raw generations, and scores them via eval/score.py. This measures a model's innate
biological prior + instruction-following.

Backends:
  - hf:      local HuggingFace causal LM (transformers) — supports text; vision via
             AutoProcessor when the model is multimodal (e.g. OmniGene-4-MM).
  - openai:  OpenAI-compatible chat API (Qwen-3 / Gemini via DashScope, etc.)
  - oracle:  debug — echoes gold answers (sanity).

Output: results/<model>__<task>.zeroshot.json with {predictions, score}.

This is intentionally dependency-light; the heavy model load only happens for backend=hf.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402


def load_test(path: str):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("split") == "test":
                rows.append(r)
    return rows


# ---- backends ----------------------------------------------------------------

def gen_oracle(rows, **kw):
    return {r["id"]: r["answer_short"] for r in rows}


def gen_hf(rows, model_path, max_new_tokens=64, batch_size=8, device="cuda", **kw):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # sm_120 / grouped-mm guard used across this project's MoE scripts
    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **k: False
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    ).eval()

    preds = {}
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        prompts = []
        for r in chunk:
            msgs = [m for m in r["messages"] if m["role"] != "assistant"]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = msgs[-1]["content"]
            prompts.append(text)
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for r, o, inp in zip(chunk, out, enc["input_ids"]):
            gen = tok.decode(o[len(inp):], skip_special_tokens=True)
            preds[r["id"]] = gen.strip()
        if (i // batch_size) % 10 == 0:
            print(f"  [hf] {i+len(chunk)}/{len(rows)}", flush=True)
    return preds


def gen_openai(rows, model_name, api_key=None, base_url=None, **kw):
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("API_KEY"),
                    base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
    preds = {}
    for j, r in enumerate(rows):
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        # vision rows: skip if API can't take local images here (kept text-only)
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
                temperature=0, max_tokens=128,
            )
            preds[r["id"]] = resp.choices[0].message.content.strip()
        except Exception as e:
            preds[r["id"]] = f"[error] {e}"
        if j % 50 == 0:
            print(f"  [openai] {j}/{len(rows)}", flush=True)
    return preds


BACKENDS = {"oracle": gen_oracle, "hf": gen_hf, "openai": gen_openai}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True, help="path to a task jsonl")
    ap.add_argument("--backend", choices=list(BACKENDS), default="oracle")
    ap.add_argument("--model", default="oracle", help="model path or API model name")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="cap test rows (debug)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    rows = load_test(a.task_file)
    if a.limit:
        rows = rows[:a.limit]
    print(f"[zeroshot] {a.task_file}: {len(rows)} test rows | backend={a.backend} model={a.model}")

    preds = BACKENDS[a.backend](
        rows, model_path=a.model, model_name=a.model,
        max_new_tokens=a.max_new_tokens, batch_size=a.batch_size,
    )
    res = score_task(a.task_file, preds)
    res["mode"] = "zeroshot"
    res["model"] = a.model
    res["task"] = os.path.basename(a.task_file).replace(".jsonl", "")

    os.makedirs(a.out_dir, exist_ok=True)
    tag = os.path.basename(a.model).replace("/", "_")
    out = os.path.join(a.out_dir, f"{tag}__{res['task']}.zeroshot.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"result": res, "predictions": preds}, fh, ensure_ascii=False, indent=2)
    print(f"[zeroshot] {res}  -> {out}")


if __name__ == "__main__":
    main()
