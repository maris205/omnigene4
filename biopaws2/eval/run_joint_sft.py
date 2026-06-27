"""BioPAWS-2 Mode B-joint: ONE multi-task SFT over the whole mixture, eval on every task.

This is the mode that demonstrates the generalist advantage (docs/generalist_vs_specialist.md):
a single chat model is LoRA-SFT'd on the concatenated train splits of ALL task files, then
evaluated separately on each task's test split. The output is a single "generalist row" plus
per-task scores — directly comparable to the N task-specific heads a PLM would require.

Only the chat / generative paradigm can do this; a classification head cannot answer
multiple task types from one set of weights. That asymmetry is the point.

Usage:
  python eval/run_joint_sft.py --base <model> --tasks data/*.jsonl --epochs 1 --out-dir results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402


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


def to_chat_text(tok, messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False)
    except Exception:
        return "\n".join(f"### {m['role']}:\n{m['content']}" for m in messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="task jsonl files or globs (text-only tasks for a text LLM)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--per-task-cap", type=int, default=None,
                    help="cap train rows per task for balance")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--unfreeze-router", action="store_true")
    ap.add_argument("--tag", default=None, help="model tag for output")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              BitsAndBytesConfig, TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **k: False
    except Exception:
        pass

    # expand globs
    task_files = []
    for t in a.tasks:
        task_files.extend(sorted(glob.glob(t)) if any(c in t for c in "*?[") else [t])
    task_files = [t for t in task_files if t.endswith(".jsonl")]
    print(f"[joint] {len(task_files)} tasks: {[os.path.basename(t) for t in task_files]}")

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- build joint train mixture ---
    mix = []
    per_task_counts = {}
    for tf in task_files:
        rows = load_split(tf, "train")
        if a.per_task_cap:
            rows = rows[:a.per_task_cap]
        per_task_counts[os.path.basename(tf)] = len(rows)
        for r in rows:
            mix.append({"text": to_chat_text(tok, r["messages"])})
    print(f"[joint] joint train mixture: {len(mix)} rows | per-task: {per_task_counts}")
    ds = Dataset.from_list(mix)

    quant = None
    if a.use_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant, trust_remote_code=True)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    if a.unfreeze_router:
        targets += ["router", "router.proj"]
    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05,
                      target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tag = a.tag or os.path.basename(a.base)
    ckpt = os.path.join(a.out_dir, f"joint_sft_{tag}")
    targs = TrainingArguments(
        output_dir=ckpt, num_train_epochs=a.epochs, per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=8, learning_rate=a.lr, bf16=True, logging_steps=25,
        save_strategy="no", report_to=[], warmup_ratio=0.03, lr_scheduler_type="cosine")
    trainer = SFTTrainer(model=model, args=targs, train_dataset=ds,
                         dataset_text_field="text", max_seq_length=a.max_seq_len, tokenizer=tok)
    trainer.train()
    model.eval()

    # --- eval the ONE model on every task's test split ---
    all_results = {}
    for tf in task_files:
        task = os.path.basename(tf).replace(".jsonl", "")
        test = load_split(tf, "test")
        preds = {}
        for i in range(0, len(test), a.bsz):
            chunk = test[i:i + a.bsz]
            prompts = [tok.apply_chat_template(
                [m for m in r["messages"] if m["role"] != "assistant"],
                tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_seq_len).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for r, o, inp in zip(chunk, out, enc["input_ids"]):
                preds[r["id"]] = tok.decode(o[len(inp):], skip_special_tokens=True).strip()
        res = score_task(tf, preds)
        res.update({"mode": "joint_sft", "model": tag, "task": task, "n_models_needed": 1})
        all_results[task] = res
        json.dump({"result": res, "predictions": preds},
                  open(os.path.join(a.out_dir, f"{tag}__{task}.joint.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  [joint-eval] {task}: {res}")

    # generalist summary
    scores = [r["score"] for r in all_results.values()]
    summary = {"model": tag, "mode": "joint_sft", "n_tasks": len(scores),
               "n_models_needed": 1, "generalist_macro": round(sum(scores)/len(scores), 4)}
    json.dump(summary, open(os.path.join(a.out_dir, f"{tag}__GENERALIST.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[joint] GENERALIST: {summary}")


if __name__ == "__main__":
    main()
