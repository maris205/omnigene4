# The Generalist-vs-Specialist Argument (BioPAWS-2 core claim)

This is the central scientific argument BioPAWS-2 is built to make measurable. It is the
reason the benchmark exists in instruction-tuning QA format.

## The structural asymmetry

| Property | Specialized PLM (ESM-2 / ProtT5 / DNABERT-2 + head) | Chat SFT model (OmniGene / LLM) |
|---|---|---|
| Tasks answerable after **one** training | **1** (one head ⇒ one task) | **N** (one multi-task SFT ⇒ all subtasks) |
| To cover all 45 BioPAWS-2 subtasks | train ~45 separate heads/models | **1** model, **1** SFT run |
| Adding a new task | train a new head from scratch | add examples to the SFT mix (or already zero-shot) |
| Deployment footprint | N models / N heads | 1 model |
| Interface | bespoke I/O per task | uniform chat → **drop-in for agent systems** |

A classification head is a **single-task specialist**. A chat model fine-tuned on the
BioPAWS-2 mixture is a **generalist**: the same weights answer homology, EC number,
promoter detection, variant effect, molecule captioning — switching task by instruction
alone.

## What "fair" therefore means here

The naive question "does the chat model beat the head on task X by 0.5 points?" is the
**wrong** comparison, because it ignores that the head can *only* do task X. The right
framing:

> **A generalist chat model that is not substantially worse than each task-specific head,
> while covering all N tasks with one model and one training run, is the better system.**

So BioPAWS-2 reports two things together:
1. **Per-task parity gap**: `chat_score − best_specialist_head_score` per task. The claim is
   this gap is small (near zero or modestly negative) — "not much worse than the head".
2. **Coverage cost**: how many models / trainings each paradigm needs to cover the suite.
   Specialist = N; generalist = 1.

The generalist wins not by topping every cell, but by *near-parity at 1/N the model count
and a unified interface*.

## Evaluation modes that make this measurable

- **Mode A** — zero-shot QA (LLM only; innate prior).
- **Mode B-single** — per-task training: PLM trains one head per task; LLM LoRA-SFTs per
  task. Apples-to-apples *per task*.
- **Mode B-joint** *(the differentiator)* — **one** multi-task SFT over the *entire*
  BioPAWS-2 train mixture, then evaluate that single model on *every* test split. Only the
  chat paradigm can do this. Reported as `joint_acc` per task + a single "generalist row".

## Leaderboard columns added

```
model · paradigm · #models_needed · #trainings · per-task scores · generalist_macro · parity_gap_vs_head
```

- `#models_needed`: 1 for a joint chat model, N for per-task heads.
- `generalist_macro`: macro score of ONE chat model across all families after joint SFT.
- `parity_gap_vs_head`: mean(chat_per_task − best_head_per_task). Target: small.

## One-line for the paper

> *On BioPAWS-2, a single instruction-tuned chat model matches task-specific
> protein/DNA-language-model heads to within a few points on each task while covering all
> 45 subtasks with one model and one training run — quantifying, for the first time on a
> unified axis, the generalist advantage of the chat paradigm over the classification-head
> paradigm, and yielding an interface that plugs directly into downstream agent systems.*
