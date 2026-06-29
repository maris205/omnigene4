"""Control experiment for the BioPAWS-2 review: raw Gemma-4-base + identical joint-SFT.

Isolates the contribution of OmniGene's adaptation stack (bio vocab + bio-CPT) by running
the SAME BioPAWS-2 joint multi-task SFT recipe on the ORIGINAL google/gemma-4-26b-a4b base
(262144 vocab, NO bio-CPT, NO vocab extension), then evaluating on every task's test split.

Matched to run_omnigene_joint.py: same task mixture, same per-task cap, same LoRA
(r=64, alpha=128, dropout 0.05, same target modules), same epochs/lr/grad-accum, same
Alpaca prompt, same greedy decode. The ONLY differences are: (a) initialization (plain base,
no bio-CPT), (b) tokenizer (original 262144 vocab), (c) fresh LoRA via get_peft_model rather
than continuing the stage2 adapter, (d) no mm_token_type_ids (plain text base, no vision tower).

Results-to-claims (pre-committed, see REVIEW_biopaws2_gpt54.md):
  - Gemma+SFT >> OmniGene+SFT  -> domain CPT does not auto-help joint QA; pivot to benchmark-first
  - Gemma+SFT ~= OmniGene+SFT  -> BioPAWS-2 is a useful SFT resource; gains mostly from supervision
  - OmniGene+SFT >> Gemma+SFT  -> OmniGene adaptation stack helps (CPT-alone needs vocab-only ablation)

Usage:
  python eval/run_gemma_base_joint.py --tasks data/protein_homology_std.jsonl ... \
      --epochs 1 --per-task-cap 6000 --lr 1e-4 --tag Gemma4-base-joint
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402

BASE = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-base"


def alpaca(instr, answer=None):
    p = f"### Instruction:\n{instr}\n\n### Answer:\n"
    return p + answer if answer is not None else p


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
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-task-cap", type=int, default=6000)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--tag", default="Gemma4-base-joint")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **k: False
    except Exception:
        pass

    task_files = []
    for t in a.tasks:
        task_files.extend(sorted(glob.glob(t)) if any(c in t for c in "*?[") else [t])
    task_files = sorted(set(t for t in task_files if t.endswith(".jsonl")))
    print(f"[gemma-joint] {len(task_files)} task files | base={BASE}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)

    # fresh LoRA from scratch (no pre-existing adapter) — matched target modules
    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                      'gate_proj', 'up_proj', 'down_proj'],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # --- build joint mixture (identical logic to OmniGene joint) ---
    mix, per_task = [], {}
    for tf in task_files:
        rows = load_split(tf, "train")[:a.per_task_cap]
        per_task[os.path.basename(tf)] = len(rows)
        for r in rows:
            instr = r["messages"][0]["content"]
            ans = r["messages"][-1]["content"]
            full = alpaca(instr, ans) + tok.eos_token
            prompt = alpaca(instr)
            ids = tok(full, truncation=True, max_length=a.max_seq_len)["input_ids"]
            plen = len(tok(prompt, truncation=True, max_length=a.max_seq_len)["input_ids"])
            labels = ([-100] * min(plen, len(ids)) + ids[min(plen, len(ids)):])[:len(ids)]
            mix.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    print(f"[gemma-joint] mixture {len(mix)} rows | per-task {per_task}", flush=True)
    ds = Dataset.from_list(mix)

    class Collator:
        def __call__(self, feats):
            maxlen = max(len(f["input_ids"]) for f in feats)
            pad = tok.pad_token_id
            b = {"input_ids": [], "attention_mask": [], "labels": []}
            for f in feats:
                n = maxlen - len(f["input_ids"])
                b["input_ids"].append(f["input_ids"] + [pad] * n)
                b["attention_mask"].append(f["attention_mask"] + [0] * n)
                b["labels"].append(f["labels"] + [-100] * n)
            return {k: torch.tensor(v) for k, v in b.items()}

    targs = TrainingArguments(
        output_dir=os.path.join(a.out_dir, f"joint_{a.tag}"),
        num_train_epochs=a.epochs, per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, bf16=True,
        logging_steps=25, save_strategy="no", report_to=[], warmup_ratio=0.03,
        lr_scheduler_type="cosine")
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator()).train()
    model.eval()

    # --- evaluate the ONE model on every task's test split ---
    eos = tok.eos_token_id
    pad = tok.pad_token_id or 0
    all_scores = {}
    for tf in task_files:
        task = os.path.basename(tf).replace(".jsonl", "")
        test = load_split(tf, "test")
        if not test:
            continue
        preds = {}
        for r in test:
            ids = tok(alpaca(r["messages"][0]["content"]), return_tensors="pt",
                      truncation=True, max_length=a.max_seq_len).input_ids.to(model.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=24, do_sample=False,
                                     eos_token_id=eos, pad_token_id=pad)
            preds[r["id"]] = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        res = score_task(tf, preds)
        res.update({"mode": "joint_sft", "model": a.tag, "task": task, "n_models_needed": 1})
        all_scores[task] = res["score"]
        json.dump({"result": res, "predictions": preds},
                  open(os.path.join(a.out_dir, f"{a.tag}__{task}.joint.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  [gemma-eval] {task}: {res['score']:.4f} ({res['metric']}, n={res['n']})",
              flush=True)

    macro = sum(all_scores.values()) / len(all_scores) if all_scores else 0
    summary = {"model": a.tag, "mode": "joint_sft", "n_tasks": len(all_scores),
               "n_models_needed": 1, "generalist_macro": round(macro, 4),
               "per_task": all_scores}
    json.dump(summary, open(os.path.join(a.out_dir, f"{a.tag}__GENERALIST.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[gemma-joint] GENERALIST macro={macro:.4f} over {len(all_scores)} tasks", flush=True)


if __name__ == "__main__":
    main()
