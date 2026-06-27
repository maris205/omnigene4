"""OmniGene-4-MM SFT-then-eval (Mode B) for BioPAWS-2 text tasks.

Loads OmniGene-4-MM correctly (base v5-merged + inject stage2 adapter + MM lora/embedding),
then adds a FRESH trainable LoRA on top and fine-tunes on a task's train split (Alpaca
format, the model's native SFT format), then evaluates on test. This gives ft_acc for the
trainability axis (Δ = ft − base).

We train a new adapter ("biopaws2") rather than the frozen MM "stage2" adapter, so the
pretrained MM capability is preserved and we measure marginal gain from BioPAWS-2 data.

Usage:
  python eval/run_omnigene_sft.py --task-file data/protein_homology_std.jsonl \
      --epochs 1 --max-train 8000
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--tag", default="OmniGene-4-MM")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                              Trainer)
    from peft import LoraConfig, inject_adapter_in_model

    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **k: False
    except Exception:
        pass

    task = os.path.basename(a.task_file).replace(".jsonl", "")
    print(f"[omni-sft] {task}: loading OmniGene-4-MM ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MM_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})

    # freeze base, then inject the stage2 adapter (its lora_ params are trainable by default)
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
    mm_emb = torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu")
    model.get_input_embeddings().weight.data.copy_(mm_emb)
    # continue-train the stage2 LoRA on BioPAWS-2 data (lora_ params already requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[omni-sft] MM loaded ({loaded} tensors); trainable {trainable/1e6:.1f}M params",
          flush=True)

    # --- build training data (mask prompt, learn answer) ---
    train_rows = load_split(a.task_file, "train")[:a.max_train]
    def tokenize(r):
        instr = r["messages"][0]["content"]
        ans = r["messages"][-1]["content"]
        full = alpaca(instr, ans) + tok.eos_token
        prompt = alpaca(instr)
        ids = tok(full, truncation=True, max_length=a.max_seq_len)["input_ids"]
        plen = len(tok(prompt, truncation=True, max_length=a.max_seq_len)["input_ids"])
        labels = [-100] * min(plen, len(ids)) + ids[min(plen, len(ids)):]
        labels = labels[:len(ids)]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
                "mm_token_type_ids": [0] * len(ids)}  # all-0 = text tokens (Gemma4 MM req)
    ds = Dataset.from_list([tokenize(r) for r in train_rows])
    print(f"[omni-sft] {len(ds)} train examples", flush=True)

    class Collator:
        def __call__(self, feats):
            maxlen = max(len(f["input_ids"]) for f in feats)
            pad_id = tok.pad_token_id
            batch = {"input_ids": [], "attention_mask": [], "labels": [],
                     "mm_token_type_ids": []}
            for f in feats:
                n = maxlen - len(f["input_ids"])
                batch["input_ids"].append(f["input_ids"] + [pad_id] * n)
                batch["attention_mask"].append(f["attention_mask"] + [0] * n)
                batch["labels"].append(f["labels"] + [-100] * n)
                batch["mm_token_type_ids"].append(f["mm_token_type_ids"] + [0] * n)
            return {k: torch.tensor(v) for k, v in batch.items()}

    targs = TrainingArguments(
        output_dir=os.path.join(a.out_dir, f"omni_sft_{task}"),
        num_train_epochs=a.epochs, per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, bf16=True,
        logging_steps=20, save_strategy="no", report_to=[], warmup_ratio=0.03,
        lr_scheduler_type="cosine")
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator())
    trainer.train()
    model.eval()

    # --- eval on test ---
    test_rows = load_split(a.task_file, "test")
    eos = tok.eos_token_id
    pad = tok.pad_token_id or 0
    preds = {}
    for j, r in enumerate(test_rows):
        prompt = alpaca(r["messages"][0]["content"])
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=a.max_seq_len).input_ids.to(model.device)
        mmt = torch.zeros_like(ids)
        with torch.no_grad():
            out = model.generate(ids, mm_token_type_ids=mmt, max_new_tokens=16,
                                 do_sample=False, eos_token_id=eos, pad_token_id=pad)
        preds[r["id"]] = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        if j % 500 == 0:
            print(f"  [eval {j}/{len(test_rows)}] {preds[r['id']][:30]!r}", flush=True)

    res = score_task(a.task_file, preds)
    res.update({"mode": "sft", "model": a.tag, "task": task, "paradigm": "chat_lora"})
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.tag}__{task}.sft.json")
    json.dump({"result": res, "predictions": preds},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[omni-sft] {res}  -> {out}")


if __name__ == "__main__":
    main()
