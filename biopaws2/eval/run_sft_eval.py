"""BioPAWS-2 Mode B: SFT-then-eval (the trainability axis).

LoRA fine-tune a base model on a task's `train` split, then evaluate on `test`, and report
ft_acc. Combined with Mode A's base_acc, the leaderboard reports Δ = ft − base, a NEW axis
quantifying how much a model improves when given BioPAWS-2 instruction data. This is the
core differentiator of BioPAWS-2: it is a training resource, not only a test set.

Lightweight LoRA SFT via peft + trl SFTTrainer, mirroring this project's existing
omnigene5 SFT scripts (same target_modules, 4-bit option, sm_120 grouped-mm guard).

Usage:
    python eval/run_sft_eval.py --task-file data/protein_homology_std.jsonl \
        --base /path/to/base_model --out-dir results --epochs 1
"""
from __future__ import annotations

import argparse
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
        # fallback simple template
        parts = []
        for m in messages:
            parts.append(f"### {m['role']}:\n{m['content']}")
        return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--base", required=True, help="base model path/name")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=None, help="cap train rows")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--unfreeze-router", action="store_true",
                    help="add gate_proj/router to LoRA targets (MoE specialization)")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              BitsAndBytesConfig, TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    # sm_120 grouped-mm guard (project-wide convention for Gemma-4 MoE)
    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **k: False
    except Exception:
        pass

    task = os.path.basename(a.task_file).replace(".jsonl", "")
    train_rows = load_split(a.task_file, "train")
    if a.max_train:
        train_rows = train_rows[:a.max_train]
    print(f"[sft] {task}: {len(train_rows)} train rows | base={a.base}")

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = [{"text": to_chat_text(tok, r["messages"])} for r in train_rows]
    ds = Dataset.from_list(texts)

    quant = None
    if a.use_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant, trust_remote_code=True,
    )

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    if a.unfreeze_router:
        # project convention: router.proj must be trainable for expert specialization
        targets += ["router", "router.proj"]
    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05,
                      target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ckpt_dir = os.path.join(a.out_dir, f"sft_{task}_{os.path.basename(a.base)}")
    targs = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bsz, gradient_accumulation_steps=4,
        learning_rate=a.lr, bf16=True, logging_steps=20, save_strategy="no",
        report_to=[], warmup_ratio=0.03, lr_scheduler_type="cosine",
    )
    trainer = SFTTrainer(model=model, args=targs, train_dataset=ds,
                         dataset_text_field="text", max_seq_length=a.max_seq_len,
                         tokenizer=tok)
    trainer.train()

    # eval on test via the zeroshot generation path (reuse run_zeroshot.gen_hf logic inline)
    model.eval()
    test_rows = load_split(a.task_file, "test")
    preds = {}
    for i in range(0, len(test_rows), a.bsz):
        chunk = test_rows[i:i + a.bsz]
        prompts = []
        for r in chunk:
            msgs = [m for m in r["messages"] if m["role"] != "assistant"]
            prompts.append(to_chat_text(tok, msgs) if False
                           else tok.apply_chat_template(msgs, tokenize=False,
                                                        add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.max_seq_len).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for r, o, inp in zip(chunk, out, enc["input_ids"]):
            preds[r["id"]] = tok.decode(o[len(inp):], skip_special_tokens=True).strip()

    res = score_task(a.task_file, preds)
    res.update({"mode": "sft", "model": a.base, "task": task, "epochs": a.epochs})
    os.makedirs(a.out_dir, exist_ok=True)
    tag = os.path.basename(a.base).replace("/", "_")
    out = os.path.join(a.out_dir, f"{tag}__{task}.sft.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"result": res, "predictions": preds}, fh, ensure_ascii=False, indent=2)
    print(f"[sft] {res}  -> {out}")


if __name__ == "__main__":
    main()
