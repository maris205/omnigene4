#!/usr/bin/env python
"""
Upload OmniGene-4 v5 (merged) + OmniGene-4-MM Stage 3 v3 (LoRA + embedding)
to Hugging Face under dnagpt/ org.

Usage:
  HF_TOKEN=hf_xxx python upload_to_hf.py [--v5-only | --mm-only | --readme-only]
"""
import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError

V5_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
MM_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-MM-stage3v3"

V5_REPO = "dnagpt/OmniGene-4-SFT-v5-merged"  # already exists; only README update if needed
MM_REPO = "dnagpt/OmniGene-4-MM-LoRA"


V5_README = """---
license: apache-2.0
language:
- en
- zh
library_name: transformers
tags:
- biology
- protein
- DNA
- mixture-of-experts
- moe
- bioinformatics
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B
---

# OmniGene-4 (v5)

A unified bio-language Mixture-of-Experts foundation model on Gemma-4-26B-A4B
(30 layers × 128 experts, top-8 routing), trained with continued pre-training
on 32.5 GB of mixed DNA + protein + 3Di + DSSP + OpenWebText, then four
cumulative SFT stages (v2 → v5). v5 adds two per-residue classification heads
(3Di and DSSP).

This repository hosts the **merged BF16 weights** (~49 GB across 11
safetensors shards). For the LoRA-only adapter (~1.7 GB), see the project
GitHub.

## Headline numbers

| Benchmark | Result | Best comparable baseline |
|---|---|---|
| BioPAWS standard homology (1k pairs) | **99.40 %** | Gemma-4-Instruct zero-shot 85 % |
| BioPAWS remote homology (500 pairs) | **82.60 %** | ESM-2 3B 51.20 %, MMseqs2 54.40 % |
| BixBench (general bio QA, T/F) | **93.66 %** | Gemma-4-Instruct 87 % |
| 3Di per-residue (chance 5 %) | **78.6 %** | — |
| DSSP per-residue (chance 12.5 %) | **100 %** | — |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("dnagpt/OmniGene-4")
model = AutoModelForCausalLM.from_pretrained(
    "dnagpt/OmniGene-4",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = '''### Instruction:
Determine if the two sequences below are structurally related (like paraphrases).

### Sequence 1:
MSRIGNKVIVLPAGVELANNDNVVTVKGPKGELTREFSKDIEIRVEGTEVTLHRPNDSKEMKTIHGTTRALL

### Sequence 2:
MSRIGNKVIVLPAGVELANNDNVVTVKGPKGELTREFSKDIEIRVEGTEVTLHRPNDSKEMKTIHGTTRALL

### Answer:
'''

ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
out = model.generate(ids, max_new_tokens=8, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
# Expected: Homologous
```

## Multi-modal version

For the vision-aware extension (OmniGene-4-MM, four vision modalities + the
above sequence/text capabilities), see [`dnagpt/OmniGene-4-MM`](https://huggingface.co/dnagpt/OmniGene-4-MM).

## Citation

```bibtex
@article{wang2026omnigene4,
  title  = {OmniGene-4: A Unified Bio-Language MoE Model with Router-Level
            Interpretability and Modality-Invariant Transfer},
  author = {Wang, Liang},
  year   = {2026},
  note   = {Manuscript at Patterns (Cell Press). Preprint:
            bioRxiv 10.1101/2026.01.03.697478. Code:
            https://github.com/maris205/omnigene4}
}
```

## License

Code: MIT (see GitHub). Model weights: Apache 2.0 (inherited from Gemma-4 base).
"""


MM_README = """---
license: apache-2.0
language:
- en
- zh
library_name: peft
tags:
- biology
- protein
- DNA
- multimodal
- vision-language
- mixture-of-experts
- moe
- lora
- bioinformatics
pipeline_tag: image-text-to-text
base_model: dnagpt/OmniGene-4-SFT-v5-merged
---

# OmniGene-4-MM (Stage 3 v3, LoRA + embedding)

Multi-modal extension of [OmniGene-4 v5](https://huggingface.co/dnagpt/OmniGene-4-SFT-v5-merged)
that adds four vision modalities (chemical-structure images, medical / pathology
imagery, charts) on top of the v5 sequence + language capability.

This repository hosts the **LoRA adapter + extended embedding** (~1.7 GB).
You need to first load the base model
[`dnagpt/OmniGene-4-SFT-v5-merged`](https://huggingface.co/dnagpt/OmniGene-4-SFT-v5-merged)
and then patch it with the artefacts here. A merged BF16 release is forthcoming
as `dnagpt/OmniGene-4-MM-merged`.

## Headline numbers

| Capability | Stage 3 v3 | v5 (text-only) |
|---|---|---|
| BioPAWS standard homology | **85.0 %** | 99.4 % |
| BioPAWS remote homology | **69.5 %** | 82.6 % |
| Vis-CheBI20 `struct_recog` | **1.00** | — |
| Vis-CheBI20 `struct_cap` | **0.96** | — |
| Cell-marker → cell-type ID (kw-overlap) | **0.95** | — |
| SMILES → physicochem descriptor (kw-overlap) | **0.91** | — |
| Protein-pair homology generation (kw-overlap) | **1.00** | — |
| Total compute | ~1.5 GPU-days (single H20) | 1.5 GPU-days |

## Files

| File | Size | What it is |
|---|---|---|
| `lora_weights.pt` | 160 MB | LoRA adapter state-dict (r=64, α=128, on q/k/v/o, gate/up/down, router.proj) |
| `embedding_weights.pt` | 1.6 GB | Extended embedding table (290,172 × 2,816, BF16) |
| `tokenizer.json` + `tokenizer_config.json` | 37 MB | Tokenizer with 28,028 biological tokens |
| `processor_config.json` | 2 KB | Multimodal processor configuration |
| `chat_template.jinja` | 16 KB | Chat template |
| `meta.json` | 0.3 KB | Training hyperparameters |

## Loading

```python
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, inject_adapter_in_model
from huggingface_hub import hf_hub_download

# 1. Load base
BASE = "dnagpt/OmniGene-4-SFT-v5-merged"
ADAPTER = "dnagpt/OmniGene-4-MM-LoRA"

tok = AutoTokenizer.from_pretrained(ADAPTER)
proc = AutoProcessor.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto",
)

# 2. Inject empty LoRA at the same target modules used during training
lora_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj','router.proj'],
)
inject_adapter_in_model(lora_cfg, model.model.language_model, adapter_name="stage2")

# 3. Patch in trained weights
sd = model.state_dict()
for k, v in torch.load(hf_hub_download(ADAPTER, "lora_weights.pt"), map_location="cpu").items():
    if k in sd: sd[k].copy_(v)
emb = torch.load(hf_hub_download(ADAPTER, "embedding_weights.pt"), map_location="cpu")
model.get_input_embeddings().weight.data.copy_(emb)
model.eval()
```

## Multi-modal usage

```python
from PIL import Image

img = Image.open("molecule.png").convert("RGB")
msgs = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Please list the functional groups of the molecule."},
]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inp = proc(text=text, images=[img], return_tensors="pt").to(model.device)

out = model.generate(**inp, max_new_tokens=160, do_sample=False)
print(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training pipeline

Three-stage LoRA fine-tuning starting from the v5 merged checkpoint:

1. **Stage 1** (~0.4 GPU-days): vision-only warmup, 10K steps, LR 5e-5
2. **Stage 2** (~1.0 GPU-days): mixed text + vision, 6K steps, LR 5e-6
3. **Stage 3 v3** (~0.5 GPU-days): heavy-homology with frozen embedding,
   3K steps, LR 2e-5

See [scripts in the GitHub repository](https://github.com/maris205/omnigene4/tree/main/omnigene5/scripts)
for complete reproducibility.

## Citation

```bibtex
@article{wang2026omnigene4,
  title  = {OmniGene-4: A Unified Bio-Language MoE Model with Router-Level
            Interpretability and Modality-Invariant Transfer},
  author = {Wang, Liang},
  year   = {2026},
  note   = {Manuscript at Patterns (Cell Press). Preprint:
            bioRxiv 10.1101/2026.01.03.697478. Code:
            https://github.com/maris205/omnigene4}
}
```

## License

Code: MIT (see GitHub). Model weights: Apache 2.0 (inherited from Gemma-4 base).
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-only", action="store_true")
    parser.add_argument("--mm-only", action="store_true")
    parser.add_argument("--readme-only", action="store_true",
                        help="Only push README.md (no model files)")
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    do_v5 = not args.mm_only
    do_mm = not args.v5_only

    if do_v5:
        print(f"\n=== Uploading {V5_REPO} ===")
        # README first (also creates repo)
        with open("/tmp/v5_readme.md", "w") as f: f.write(V5_README)
        try:
            create_repo(V5_REPO, token=token, exist_ok=True)
            print("  repo OK")
        except Exception as e:
            print(f"  repo creation: {e}")
        api.upload_file(
            path_or_fileobj="/tmp/v5_readme.md",
            path_in_repo="README.md",
            repo_id=V5_REPO,
            token=token,
            commit_message="Add README with usage and benchmarks",
        )
        print(f"  README pushed")

        if not args.readme_only:
            print(f"  uploading folder {V5_DIR} (this is large; ~49 GB)")
            api.upload_folder(
                folder_path=V5_DIR,
                repo_id=V5_REPO,
                token=token,
                commit_message="Upload OmniGene-4 v5 merged BF16 weights",
                ignore_patterns=["*.log", "*.json.tmp", "*.lock", "__pycache__/*",
                                 ".ipynb_checkpoints/*", "README.md"],
            )
            print(f"  v5 done")

    if do_mm:
        print(f"\n=== Uploading {MM_REPO} ===")
        with open("/tmp/mm_readme.md", "w") as f: f.write(MM_README)
        try:
            create_repo(MM_REPO, token=token, exist_ok=True)
            print("  repo OK")
        except Exception as e:
            print(f"  repo creation: {e}")
        api.upload_file(
            path_or_fileobj="/tmp/mm_readme.md",
            path_in_repo="README.md",
            repo_id=MM_REPO,
            token=token,
            commit_message="Add README with usage and benchmarks",
        )
        print(f"  README pushed")

        if not args.readme_only:
            print(f"  uploading folder {MM_DIR}")
            api.upload_folder(
                folder_path=MM_DIR,
                repo_id=MM_REPO,
                token=token,
                commit_message="Upload OmniGene-4-MM Stage 3 v3 LoRA + embedding",
                ignore_patterns=["*.log", "*.json.tmp", "*.lock", "__pycache__/*",
                                 ".ipynb_checkpoints/*", "demo.log", "eval.log",
                                 "train.log", "README.md"],
            )
            print(f"  MM done")

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
