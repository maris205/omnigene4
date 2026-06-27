"""OmniGene-4-MM zero-shot runner for BioPAWS-2 text tasks (F1 etc.).

Loads the model EXACTLY as the proven omnigene5/scripts/92-eval_stage3v3.py does:
  base = OmniGene-4-v5-merged  +  inject_adapter("stage2")  +  load stage3v3
  lora_weights.pt + embedding_weights.pt.

Uses the Alpaca prompt format the model was SFT'd on (NOT chat_template — that produces
garbage). Scores via eval/score.py. This is Mode A (zero-shot QA) for our own model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3"


def make_alpaca_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Answer:\n"


def load_split(path, split):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("split") == split:
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--tag", default="OmniGene-4-MM")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, inject_adapter_in_model

    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **k: False
    except Exception:
        pass

    print(f"[omnigene] loading base {BASE_MODEL} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MM_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
    lora_cfg = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj', 'router.proj'])
    inject_adapter_in_model(lora_cfg, model.model.language_model, adapter_name="stage2")
    ms = model.state_dict()
    mm_lora = torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu")
    loaded = 0
    for k, v in mm_lora.items():
        if k in ms:
            ms[k].copy_(v)
            loaded += 1
    print(f"[omnigene] loaded {loaded} LoRA tensors", flush=True)
    assert loaded > 400, f"LoRA load failed: {loaded}"
    mm_emb = torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu")
    model.get_input_embeddings().weight.data.copy_(mm_emb)
    model.eval()
    print(f"[omnigene] GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    task = os.path.basename(a.task_file).replace(".jsonl", "")
    rows = load_split(a.task_file, "test")
    if a.limit:
        rows = rows[:a.limit]
    print(f"[omnigene] {task}: {len(rows)} test rows", flush=True)

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id or 0
    preds = {}
    for j, r in enumerate(rows):
        instr = r["messages"][0]["content"]
        prompt = make_alpaca_prompt(instr)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=2048).input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 eos_token_id=eos, pad_token_id=pad)
        gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        preds[r["id"]] = gen
        if j % 200 == 0:
            print(f"  [{j}/{len(rows)}] e.g. {gen[:40]!r}", flush=True)

    res = score_task(a.task_file, preds)
    res.update({"mode": "zeroshot", "model": a.tag, "task": task})
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.tag}__{task}.zeroshot.json")
    json.dump({"result": res, "predictions": preds},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[omnigene] {res}  -> {out}")


if __name__ == "__main__":
    main()
