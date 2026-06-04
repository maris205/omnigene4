---
license: apache-2.0
language:
- en
- zh
library_name: transformers
tags:
- biology
- protein
- DNA
- multimodal
- vision-language
- mixture-of-experts
- moe
- bioinformatics
pipeline_tag: image-text-to-text
base_model: dnagpt/OmniGene-4-SFT-v5-merged
---

# OmniGene-4-MM (merged BF16)

Stand-alone multi-modal extension of [OmniGene-4 v5](https://huggingface.co/dnagpt/OmniGene-4-SFT-v5-merged)
with the Stage 3 v3 LoRA adapter and extended embedding **already merged
into the base** weights. Load with a single `from_pretrained` and run.

For the LoRA-only release (~1.7 GB, requires the v5 base separately) see
[`dnagpt/OmniGene-4-MM-LoRA`](https://huggingface.co/dnagpt/OmniGene-4-MM-LoRA).

## Headline numbers

| Capability | Value | v5 baseline (text-only) |
|---|---|---|
| BioPAWS standard homology | **85.0 %** | 99.4 % |
| BioPAWS remote homology | **69.5 %** | 82.6 % |
| Vis-CheBI20 `struct_recog` | **1.00** | — |
| Vis-CheBI20 `struct_cap` | **0.96** | — |
| Cell-marker → cell-type ID (kw-overlap) | **0.95** | — |
| SMILES → physico-chem descriptor (kw-overlap) | **0.91** | — |
| Protein-pair homology generation (kw-overlap) | **1.00** | — |
| Total fine-tuning compute | **~1.5 GPU-days** (single H20) | 1.5 GPU-days |

OmniGene-4-MM beats classical alignment tools (MMseqs2 +15 pp, DIAMOND +16 pp)
and dense protein language models (ESM-2 3B +18 pp) on remote homology — at
roughly four orders of magnitude less compute than recent specialized MoE
bio-models such as AIDO.Protein.

## Loading and inference

```python
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from PIL import Image

REPO = "dnagpt/OmniGene-4-MM-merged"

tok = AutoTokenizer.from_pretrained(REPO)
proc = AutoProcessor.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(
    REPO, torch_dtype=torch.bfloat16, device_map="auto",
)
model.eval()

# --- Image: list functional groups of a chemical structure ---
img = Image.open("molecule.png").convert("RGB")
msgs = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Please list the functional groups of the molecule."},
]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inp = proc(text=text, images=[img], return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=160, do_sample=False)
print(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))

# --- Text: protein homology decision ---
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

## Architecture

- Base: Gemma-4-26B-A4B (30 transformer layers · 128 experts/layer · top-8
  routing · ~3.8 B active parameters per token)
- Vision tower: native Gemma-4 ViT (27 layers, 1152 hidden, patch 16, 2520
  visual patches per image)
- Vocabulary: 28,028 biological tokens injected on top of the Gemma-4 vocab
  (DNA BPE, protein BPE, 20 Foldseek 3Di letters, 8 DSSP labels)

The merged weights are produced by applying a Stage 3 v3 LoRA adapter
(r=64, α=128, on Q/K/V/O, gate/up/down, router.proj) to the v5 merged base
and folding the LoRA contribution into each linear layer.

## Training pipeline

Four cumulative SFT stages (v2 → v5) on the text-only base, then three
multi-modal LoRA stages on top:

1. **MM Stage 1** (~0.4 GPU-days): vision-only warmup, 10 K steps, LR 5e-5
2. **MM Stage 2** (~1.0 GPU-days): mixed text + vision, 6 K steps, LR 5e-6
3. **MM Stage 3 v3** (~0.5 GPU-days): heavy-homology with frozen embedding,
   3 K steps, LR 2e-5

See the [GitHub repository](https://github.com/maris205/omnigene4/tree/main/omnigene5/scripts)
for full reproducibility.

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
