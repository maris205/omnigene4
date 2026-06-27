"""OmniGene-4-MM JOINT multi-task SFT (Mode B-joint) for BioPAWS-2.

The headline generalist experiment: ONE OmniGene-4-MM model is LoRA-SFT'd on the MIXED
train splits of all given text tasks at once, then evaluated separately on EVERY task's
test split (including tasks with NO train split — measuring cross-task generalization).

Produces a single "generalist row": one model, one training, N tasks. This is the direct
foil to the N task-specific PLM heads. Only the chat paradigm can do this.

Loads OmniGene-4-MM the proven way (base v5-merged + inject stage2 adapter + MM
lora/embedding), trains the stage2 LoRA further on the mixture (Alpaca format,
mm_token_type_ids=0 for text), then greedy-decodes each test set.

Usage:
  python eval/run_omnigene_joint.py --tasks data/lg_*.jsonl data/protein_homology_std.jsonl \
      --epochs 1 --per-task-cap 8000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.score import score_task  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3"


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
    ap.add_argument("--per-task-cap", type=int, default=8000)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--tag", default="OmniGene-4-MM-joint")
    ap.add_argument("--eval-all", action="store_true",
                    help="also evaluate tasks that had no train split")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, inject_adapter_in_model

    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **k: False
    except Exception:
        pass

    # expand globs
    task_files = []
    for t in a.tasks:
        task_files.extend(sorted(glob.glob(t)) if any(c in t for c in "*?[") else [t])
    task_files = sorted(set(t for t in task_files if t.endswith(".jsonl")))
    print(f"[joint] {len(task_files)} task files", flush=True)

    print("[joint] loading OmniGene-4-MM ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MM_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
    for p in model.parameters():
        p.requires_grad = False
    mm_cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                        'gate_proj', 'up_proj', 'down_proj', 'router.proj'])
    inject_adapter_in_model(mm_cfg, model.model.language_model, adapter_name="stage2")
    model._hf_peft_config_loaded = True
    ms = model.state_dict()
    mm_lora = torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu")
    loaded = 0
    for k, v in mm_lora.items():
        if k in ms:
            ms[k].copy_(v); loaded += 1
    assert loaded > 400, f"MM LoRA load failed: {loaded}"
    model.get_input_embeddings().weight.data.copy_(
        torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[joint] MM loaded ({loaded}); trainable {trainable/1e6:.0f}M", flush=True)

    # --- build JOINT train mixture ---
    mix = []
    per_task = {}
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
            mix.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                        "labels": labels, "mm_token_type_ids": [0] * len(ids)})
    print(f"[joint] mixture {len(mix)} rows | per-task {per_task}", flush=True)
    ds = Dataset.from_list(mix)

    class Collator:
        def __call__(self, feats):
            maxlen = max(len(f["input_ids"]) for f in feats)
            pad = tok.pad_token_id
            b = {"input_ids": [], "attention_mask": [], "labels": [], "mm_token_type_ids": []}
            for f in feats:
                n = maxlen - len(f["input_ids"])
                b["input_ids"].append(f["input_ids"] + [pad] * n)
                b["attention_mask"].append(f["attention_mask"] + [0] * n)
                b["labels"].append(f["labels"] + [-100] * n)
                b["mm_token_type_ids"].append(f["mm_token_type_ids"] + [0] * n)
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
                out = model.generate(ids, mm_token_type_ids=torch.zeros_like(ids),
                                     max_new_tokens=24, do_sample=False,
                                     eos_token_id=eos, pad_token_id=pad)
            preds[r["id"]] = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        res = score_task(tf, preds)
        res.update({"mode": "joint_sft", "model": a.tag, "task": task, "n_models_needed": 1})
        all_scores[task] = res["score"]
        json.dump({"result": res, "predictions": preds},
                  open(os.path.join(a.out_dir, f"{a.tag}__{task}.joint.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  [joint-eval] {task}: {res['score']:.4f} ({res['metric']}, n={res['n']})",
              flush=True)

    macro = sum(all_scores.values()) / len(all_scores) if all_scores else 0
    summary = {"model": a.tag, "mode": "joint_sft", "n_tasks": len(all_scores),
               "n_models_needed": 1, "generalist_macro": round(macro, 4),
               "per_task": all_scores}
    json.dump(summary, open(os.path.join(a.out_dir, f"{a.tag}__GENERALIST.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[joint] GENERALIST macro={macro:.4f} over {len(all_scores)} tasks (1 model)")


if __name__ == "__main__":
    main()
