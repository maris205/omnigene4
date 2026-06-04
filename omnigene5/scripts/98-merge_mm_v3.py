#!/usr/bin/env python
"""
98-merge_mm_v3.py

Merge OmniGene-4-MM Stage 3 v3 LoRA + extended embedding INTO the v5-merged
base, producing a stand-alone BF16 multi-modal checkpoint at
  /root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-merged/

Usage:
  python 98-merge_mm_v3.py [--no-save] [--quick-test]

Notes:
- Requires ~50 GB GPU memory (BF16 base) + ~3 GB for LoRA materialization.
- The output is a standard transformers AutoModelForCausalLM checkpoint:
  user just needs from_pretrained("dnagpt/OmniGene-4-MM-merged").
"""
import os
import argparse
import json
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM,
)
from peft import LoraConfig, inject_adapter_in_model

BASE_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR   = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3"
OUT_DIR  = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-merged"

ADAPTER_NAME = "stage2"  # state-dict key suffix used by Stage 3 v3
LORA_R, LORA_ALPHA, LORA_DROPOUT = 64, 128, 0.05
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                  'gate_proj', 'up_proj', 'down_proj', 'router.proj']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true",
                        help="Run merge but skip save_pretrained (sanity check only)")
    parser.add_argument("--quick-test", action="store_true",
                        help="Skip merge entirely, just verify load+adapter inject")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Merging {MM_DIR}\n  into base {BASE_DIR}\n  output to {OUT_DIR}")
    print("=" * 60)

    # Load tokenizer + processor from MM dir (carries vocab + multimodal config)
    print("\n[1/5] Loading tokenizer + processor from MM dir...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MM_DIR)
    processor = AutoProcessor.from_pretrained(MM_DIR)
    print(f"  vocab size: {len(tokenizer)}")

    # Load base BF16 to GPU
    print(f"\n[2/5] Loading base ({BASE_DIR}) to GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    print(f"  GPU mem after base: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Inject LoRA adapter
    print(f"\n[3/5] Injecting LoRA r={LORA_R}/{LORA_ALPHA}, name='{ADAPTER_NAME}'...",
          flush=True)
    peft_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=TARGET_MODULES,
    )
    inject_adapter_in_model(peft_cfg, model.model.language_model,
                            adapter_name=ADAPTER_NAME)

    # Load MM LoRA weights into the named adapter
    print("[4/5] Loading MM v3 LoRA weights...", flush=True)
    sd = model.state_dict()
    mm_lora = torch.load(f"{MM_DIR}/lora_weights.pt", map_location="cpu")
    loaded, missing = 0, []
    for k, v in mm_lora.items():
        if k in sd:
            sd[k].copy_(v); loaded += 1
        else:
            missing.append(k)
    print(f"  loaded {loaded} LoRA tensors")
    if missing:
        print(f"  WARNING: {len(missing)} unmatched keys, sample: {missing[:3]}")
    assert loaded > 400, f"Only {loaded} LoRA tensors loaded — not safe to merge"

    # Load MM extended embedding
    print("  loading MM embedding...", flush=True)
    mm_emb = torch.load(f"{MM_DIR}/embedding_weights.pt", map_location="cpu")
    model.get_input_embeddings().weight.data.copy_(mm_emb)
    print(f"  embedding shape: {mm_emb.shape}")

    print(f"  GPU mem after MM patch: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    if args.quick_test:
        print("\n[QUICK TEST DONE] Skipping merge & save.")
        return

    # Merge LoRA into base weights
    print("\n[5/5] Merging LoRA into base weights (this rewrites Q/K/V/O, "
          "gate/up/down, router.proj for every layer)...", flush=True)
    # peft `inject_adapter_in_model` does not give us a PeftModel directly,
    # so we walk the modules and call `merge` on each LoRA layer.
    from peft.tuners.lora.layer import LoraLayer
    n_merged = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge(adapter_names=[ADAPTER_NAME])
            n_merged += 1
    print(f"  merged {n_merged} LoRA layers into base")
    print(f"  GPU mem after merge: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    if args.no_save:
        print("\n[--no-save] Sanity merge complete; skipping save_pretrained.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nSaving merged BF16 model to {OUT_DIR}...")
    # save_pretrained on the underlying base model (LoRA layers are now folded
    # into the linear weights, so the LoRA structure is irrelevant for inference)
    model.save_pretrained(OUT_DIR, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "merge_meta.json"), "w") as f:
        json.dump({
            "base":  BASE_DIR,
            "mm_lora": MM_DIR,
            "adapter_name": ADAPTER_NAME,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "n_lora_tensors_loaded": loaded,
            "n_layers_merged": n_merged,
            "mm_emb_shape": list(mm_emb.shape),
        }, f, indent=2)
    print("Done.")
    print(f"\nDirectory size:")
    os.system(f"du -sh {OUT_DIR}")


if __name__ == "__main__":
    main()
