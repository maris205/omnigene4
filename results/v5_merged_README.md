# OmniGene-4-SFT-v5-merged

**Full BF16 model with CPT + SFT v2/v3/v4/v5 + dual-head architecture (3Di + DSSP classifiers)**

This is the latest and most capable OmniGene-4 model, merged into a standalone BF16 model. No need to load base Gemma-4 separately.

## 🎯 What's New in v5

OmniGene-4-SFT-v5 introduces a **dual-head architecture**:
- **Generation head** (LM): for natural language tasks (homology, BixBench, knowledge QA)
- **3Di classifier head** (per-residue, 20-class): for Foldseek 3D structural alphabet
- **DSSP classifier head** (per-residue, 8-class): for secondary structure

**Joint loss during training**: 0.5 × generation_CE + 0.5 × classification_CE

This gives the model **two independent inference paths**: chat-based generation for unrestricted language, and direct token-level classification for structured biological prediction.

## 📊 Performance

### Latest evaluation (4-bit + Alpaca prompt)

| Benchmark | Accuracy |
|---|---|
| **Standard Homology** (6,000 pairs) | **99.40%** |
| **Remote Homology** (2,000 pairs) | **82.60%** ⭐ |
| **BixBench Knowledge** (T/F) | **93.66%** |

### Multi-task generation (6 categories × 100 samples)

| Task | Score |
|---|---|
| Protein | 100.0% |
| Mol | 98.0% |
| Cell | 96.6% |
| Literature | 55.9% |
| Mutation | 11.3% |
| Structure (gen mode) | 25.9% char overlap |

### Classification heads (per-residue, NEW in v5)

| Head | Accuracy | vs chance |
|---|---|---|
| **3Di** (20-class) | **78.6%** | 15.7× above chance (5%) |
| **DSSP** (8-class) | **100.0%** | 8× above chance (12.5%) |

### Comparison vs ESM-2 (650M) on identical 500-pair remote homology

OmniGene-4 v5: **82.60%** | ESM-2: 50.50% | Gap: **+32.1 percentage points**

## Quick Start

### Generation tasks (BF16, 49 GB GPU)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "dnagpt/OmniGene-4-SFT-v5-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("dnagpt/OmniGene-4-SFT-v5-merged")

# Example: Protein homology detection
prompt = """### Instruction:
Determine if the two sequences below are structurally related (like paraphrases).

### Sequence 1:
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQ

### Sequence 2:
MKKFDRGEQVVKVKALPQAQFEEVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDG

### Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

### Classification head usage (3Di / DSSP per-residue prediction)

```python
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Load classification heads
heads_path = hf_hub_download(
    repo_id="dnagpt/OmniGene-4-SFT-v5-merged",
    filename="struct_heads.pt",
)
heads = torch.load(heads_path, map_location="cuda")

head_3di = nn.Linear(2816, 20).to(torch.bfloat16).cuda()
head_dssp = nn.Linear(2816, 8).to(torch.bfloat16).cuda()
head_3di.load_state_dict(heads["head_3di"])
head_dssp.load_state_dict(heads["head_dssp"])

# Predict 3Di / DSSP per-residue
TDI_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
DSSP_ALPHABET = list("HBEGITSC")

prompt = "### Instruction:\nPredict 3Di for: MKTAYIAK\n\n### Answer:\n"
ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
    out = model(ids, output_hidden_states=True)
    hidden = out.hidden_states[-1][0]  # [seq_len, 2816]

    # For each position after prompt, predict 3Di or DSSP
    for i in range(ids.shape[1] - 5, ids.shape[1]):
        logits_3di = head_3di(hidden[i])
        pred_3di = TDI_ALPHABET[logits_3di.argmax().item()]
        print(f"Position {i}: 3Di={pred_3di}")
```

### 4-bit quantization (16 GB GPU)

```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "dnagpt/OmniGene-4-SFT-v5-merged",
    quantization_config=bnb,
    device_map="auto",
)
# struct_heads.pt loaded same as above
```

## 🏗️ Training Lineage

OmniGene-4 follows a cumulative training pipeline. Each stage initializes from the previous LoRA + embedding:

```
Gemma-4-26B-A4B-Instruct-bio (vocab-extended)
    ↓ CPT v2 (32.5 GB mixed corpus, 0.6 epoch, 100 GPU-h)
OmniGene-4-CPT-v2
    ↓ Bio-SFT v2 (179K instructions, 1 epoch, 11.8 GPU-h)
OmniGene-4-SFT-v2
    ↓ Bio-SFT v3 (+20K remote homology pairs, 1 epoch, 13.2 GPU-h)
OmniGene-4-SFT-v3
    ↓ Bio-SFT v4 (Alpaca template + loss masking + task reweighting, 4257 steps, 30 GPU-h)
OmniGene-4-SFT-v4
    ↓ Bio-SFT v5 (dual-head: gen + 3Di + DSSP classifiers, 1350 steps, 5 GPU-h)
OmniGene-4-SFT-v5  ← YOU ARE HERE
```

This merged model contains **all changes** from CPT through v5.

## 🧬 Architecture

- **Base**: Gemma-4-26B-A4B-Instruct
- **Layers**: 30 transformer layers
- **MoE**: 128 experts per layer, top-8 routing
- **Active params**: ~3.8B per token
- **Total params**: ~26B
- **Vocabulary**: 290,048 tokens (262,020 original + 28,028 bio tokens)
- **Bio tokens**: DNA BPE (20k) + Protein BPE (8k) + 3Di (20) + DSSP (8) + control

## 📁 Files

- `model-*.safetensors` (49 GB, 11 shards) — full BF16 weights
- `struct_heads.pt` (157 KB) — 3Di + DSSP classification heads
- `tokenizer.json` (36 MB) — extended tokenizer
- `bio_sft_v5_meta.json` — training metadata
- `chat_template.jinja` — chat template

## 🔬 Research Findings

OmniGene-4 v5 demonstrates a **Pareto-optimal** improvement over previous versions:
- ✅ Maintains v4's Remote Homology breakthrough (82.6%)
- ✅ Restores v3's BixBench performance (93.66%)
- ✅ Adds new token-level structured prediction (3Di 78.6%, DSSP 100%)
- ✅ No regression on any task

Key insight: **dual-head architecture** allows simultaneous chat-based generation and structured prediction without interference.

## 🔗 Related Models

- **LoRA adapter** (1.9 GB, requires base): https://huggingface.co/dnagpt/OmniGene-4-SFT-v5
- **Previous version (v3)**: https://huggingface.co/dnagpt/OmniGene-4-SFT-v3-merged
- **Previous version (v4)**: https://huggingface.co/dnagpt/OmniGene-4-SFT-v4-merged
- **CPT only**: https://huggingface.co/dnagpt/OmniGene-4-CPT-v2-merged
- **GGUF Q4_K_M** (16 GB): https://huggingface.co/dnagpt/OmniGene-4-SFT-v3-GGUF (v3 only — v5 GGUF coming)

## 📄 Paper

**bioRxiv preprint**: TBD link  
**GitHub**: https://github.com/maris205/omnigene4  

## 📚 Citation

```bibtex
@article{wang2026omnigene4,
  title={OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability},
  author={Wang, Liang},
  journal={bioRxiv},
  year={2026}
}
```

## 📜 License

Apache 2.0 (inherits from Gemma-4)

## 📧 Contact

Liang Wang (wangliang.f@gmail.com)  
School of Artificial Intelligence and Automation  
Huazhong University of Science and Technology
