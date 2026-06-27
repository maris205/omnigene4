"""OmniGene-4-MM multimodal (F9) zero-shot runner for BioPAWS-2.

Evaluates the vision+text tasks (molecule-image recognition / captioning / IUPAC) by
feeding the image through the Gemma-4 vision tower via AutoProcessor, exactly as the proven
omnigene5/scripts/92-eval_stage3v3.py gen_with_image path. Scores via eval/score.py.

Loads OmniGene-4-MM correctly (base v5-merged + inject stage2 adapter + MM lora/embedding).

Usage:
  python eval/run_omnigene_mm.py --task-file data/f9_vis_chebi20.jsonl --limit 300
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
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--tag", default="OmniGene-4-MM")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
    from peft import LoraConfig, inject_adapter_in_model

    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **k: False
    except Exception:
        pass

    print("[mm] loading OmniGene-4-MM ...", flush=True)
    processor = AutoProcessor.from_pretrained(MM_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MM_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
    mm_cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                        'gate_proj', 'up_proj', 'down_proj', 'router.proj'])
    inject_adapter_in_model(mm_cfg, model.model.language_model, adapter_name="stage2")
    ms = model.state_dict()
    mm_lora = torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu")
    loaded = 0
    for k, v in mm_lora.items():
        if k in ms:
            ms[k].copy_(v); loaded += 1
    assert loaded > 400, f"MM LoRA load failed: {loaded}"
    model.get_input_embeddings().weight.data.copy_(
        torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu"))
    model.eval()
    print(f"[mm] loaded ({loaded} tensors)", flush=True)

    task = os.path.basename(a.task_file).replace(".jsonl", "")
    rows = load_test(a.task_file, a.limit)
    print(f"[mm] {task}: {len(rows)} test rows", flush=True)

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id or 0
    preds = {}
    for j, r in enumerate(rows):
        user_text = r["messages"][0]["content"].replace("<image>", "").strip()
        img_path = r["images"][0] if r.get("images") else None
        try:
            img = Image.open(img_path).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": user_text}]}]
            text = processor.apply_chat_template(msgs, add_generation_prompt=True,
                                                 tokenize=False)
            inp = processor(text=text, images=[img], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     eos_token_id=eos, pad_token_id=pad)
            gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            preds[r["id"]] = gen.strip()
        except Exception as e:
            preds[r["id"]] = f"[error] {e}"
        if j % 50 == 0:
            print(f"  [mm {j}/{len(rows)}] {preds[r['id']][:50]!r}", flush=True)

    res = score_task(a.task_file, preds)
    res.update({"mode": "zeroshot", "model": a.tag, "task": task})
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.tag}__{task}.zeroshot.json")
    json.dump({"result": res, "predictions": preds},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[mm] {res}  -> {out}")


if __name__ == "__main__":
    main()
