#!/usr/bin/env python
# coding: utf-8
"""
41-train_bio_sft_v5_classifier.py

Bio-SFT v5: 基于 v4 加辅助分类 head
=================================================

设计动机:
  - v4 修了 chat-tag collapse 和 loss masking, Structure overlap ~30%
  - 但 3Di/DSSP 是有限字符表 (20 / 8)，纯 generation 浪费容量
  - 加一个辅助 per-residue classification head 直接监督结构 token

架构:
  base model (frozen) -> hidden_state[-1, :, :]
                              ↓
                          LoRA 主干 -> generation head (LM)
                              ↓
                          额外的 classifier:
                              hidden -> 3Di_head (20-class)
                              hidden -> DSSP_head (8-class)

Loss:
  L = α * generation_CE + β * classification_CE
  α=0.5, β=0.5 (平衡两个目标)

只训练 Structure 任务，其他任务保持 v4 不动。
预计 15-20 GPU 小时单卡。
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME
from peft import LoraConfig, inject_adapter_in_model

BASE_MODEL = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
SFT_V4_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v4-sft"   # waits for v4
TRAIN_FILE = "/root/autodl-fs/omnigene_v2/sft_data/train/omnigene_sft_v1_train_with_remote.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-sft-classifier"
CHECKPOINT_DIR = "/root/autodl-fs/checkpoints_sft_v5"

MAX_LENGTH = 1536
HIDDEN_SIZE = 2816  # Gemma-4 hidden size
N_3DI = 20
N_DSSP = 8
ALPHA = 0.5  # gen loss weight
BETA = 0.5   # cls loss weight

# 3Di 20 字符 vocabulary (Foldseek)
TDI_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")  # standard 20 letters
TDI_TO_ID = {c: i for i, c in enumerate(TDI_ALPHABET)}

# DSSP 8 secondary structure types
DSSP_ALPHABET = list("HBEGITSC")  # H E G I T B S C
DSSP_TO_ID = {c: i for i, c in enumerate(DSSP_ALPHABET)}


def task_of(instr):
    s = instr.lower()
    if "3di" in s or "foldseek" in s:
        return "3Di"
    if "secondary structure" in s or "dssp" in s:
        return "DSSP"
    return "Other"


def extract_structure_string(output):
    """从 SFT 输出中抽出 <SEQ_3Di>...</SEQ_3Di> 或 <SEQ_2D>...</SEQ_2D> 内容"""
    m = re.search(r'<SEQ_3Di>([A-Z]+)</SEQ_3Di>', output)
    if m:
        return ("3Di", m.group(1))
    m = re.search(r'<SEQ_2D>([A-Z]+)</SEQ_2D>', output)
    if m:
        return ("DSSP", m.group(1))
    return (None, None)


print("Loading & filtering Structure data for v5...", flush=True)
struct_records = []
with open(TRAIN_FILE) as f:
    for line in f:
        r = json.loads(line)
        task = task_of(r.get("instruction", ""))
        if task in ("3Di", "DSSP"):
            kind, seq = extract_structure_string(r["output"])
            if kind:
                r["__struct_kind__"] = kind
                r["__struct_seq__"] = seq
                struct_records.append(r)

print(f"  Structure records: {len(struct_records)}")
counts = {"3Di": sum(1 for r in struct_records if r["__struct_kind__"] == "3Di"),
          "DSSP": sum(1 for r in struct_records if r["__struct_kind__"] == "DSSP")}
print(f"    {counts}")

dataset = Dataset.from_list(struct_records)


# ================= Custom model with classification heads =================
class StructAuxModel(nn.Module):
    """包装 base model + 两个辅助 classification head."""
    def __init__(self, base_model, hidden_size, n_3di, n_dssp):
        super().__init__()
        self.base = base_model
        # 在 hidden_state 上接两个 head
        self.head_3di = nn.Linear(hidden_size, n_3di)
        self.head_dssp = nn.Linear(hidden_size, n_dssp)

    def forward(self, input_ids, attention_mask, labels=None,
                struct_kind=None, struct_target_ids=None,
                struct_target_positions=None, **kwargs):
        # base model forward, 获取 hidden states
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )
        gen_loss = out.loss

        # classification loss: 在 struct_target_positions 处取 hidden, 过 head
        cls_loss = torch.tensor(0.0, device=input_ids.device)
        if struct_kind is not None and struct_target_ids is not None:
            hidden = out.hidden_states[-1]  # [B, L, H]
            cls_losses = []
            for b in range(input_ids.shape[0]):
                kind = struct_kind[b]
                positions = struct_target_positions[b]  # [n_target_pos]
                target_ids = struct_target_ids[b]
                if positions.numel() == 0:
                    continue
                pos_hidden = hidden[b, positions]  # [n_target, H]
                if kind == "3Di":
                    logits = self.head_3di(pos_hidden)
                else:
                    logits = self.head_dssp(pos_hidden)
                cls_losses.append(F.cross_entropy(logits, target_ids))
            if cls_losses:
                cls_loss = torch.stack(cls_losses).mean()

        loss = ALPHA * gen_loss + BETA * cls_loss
        return {"loss": loss, "gen_loss": gen_loss, "cls_loss": cls_loss}


# ================= 加载 base + v4 LoRA =================
print("Loading model + v4 LoRA...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map={"": 0},
)
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

tokenizer = AutoTokenizer.from_pretrained(SFT_V4_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)
for p in base_model.parameters():
    p.requires_grad = False
inject_adapter_in_model(peft_config, base_model.model.language_model, adapter_name="default")
base_model._hf_peft_config_loaded = True

# Load v4 LoRA + embedding
v4_lora = torch.load(f"{SFT_V4_DIR}/lora_weights.pt", map_location="cpu")
ms = base_model.state_dict()
loaded = 0
for k, v in v4_lora.items():
    if k in ms:
        ms[k].copy_(v); loaded += 1
print(f"  Loaded {loaded} v4 LoRA tensors")

v4_embed = torch.load(f"{SFT_V4_DIR}/embedding_weights.pt", map_location="cpu")
base_model.get_input_embeddings().weight.data.copy_(v4_embed)
for p in base_model.get_input_embeddings().parameters():
    p.requires_grad = True

# 包装
model = StructAuxModel(base_model, HIDDEN_SIZE, N_3DI, N_DSSP)
model.head_3di = model.head_3di.to(torch.bfloat16).cuda()
model.head_dssp = model.head_dssp.to(torch.bfloat16).cuda()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,}")


# ================= Tokenize with classification targets =================
def tok_fn(examples):
    input_ids_list, labels_list, am_list = [], [], []
    struct_kinds, struct_target_ids_list, struct_target_pos_list = [], [], []

    for i in range(len(examples["instruction"])):
        instr = examples["instruction"][i]
        inp = examples["input"][i]
        out_str = examples["output"][i]
        kind = examples["__struct_kind__"][i]
        struct_seq = examples["__struct_seq__"][i]

        # Alpaca prompt
        if inp.strip():
            prompt = f"### Instruction:\n{instr}\n\n{inp}\n\n### Answer:\n"
        else:
            prompt = f"### Instruction:\n{instr}\n\n### Answer:\n"

        # 组装完整序列, 得到 prompt_len 用于 mask
        ans_with_eos = out_str + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH-256,
                                add_special_tokens=False).input_ids
        ans_ids = tokenizer(ans_with_eos, truncation=True, max_length=MAX_LENGTH-len(prompt_ids),
                             add_special_tokens=False).input_ids
        full_ids = prompt_ids + ans_ids
        labels = [-100] * len(prompt_ids) + ans_ids[:]
        am = [1] * len(full_ids)

        # 找出 struct_seq 中每个 char 对应的 token position
        # 简化方法: 找 <SEQ_*> token, 然后逐 char 检查后面 token
        target_alphabet = TDI_TO_ID if kind == "3Di" else DSSP_TO_ID
        target_ids = []
        target_positions = []

        # 在 ans_ids 中找 <SEQ_3Di> 的 token, 然后开始逐字符记录
        # 这里简化: 直接看 full_ids 中每个 token 解码后是否是 1 个 alphabet 字符
        for j, tid in enumerate(full_ids):
            if labels[j] == -100:
                continue  # prompt 部分跳过
            tok_str = tokenizer.decode([tid], skip_special_tokens=False)
            if len(tok_str) == 1 and tok_str.upper() in target_alphabet:
                target_ids.append(target_alphabet[tok_str.upper()])
                target_positions.append(j)

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        am_list.append(am)
        struct_kinds.append(kind)
        struct_target_ids_list.append(target_ids)
        struct_target_pos_list.append(target_positions)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": am_list,
        "struct_kind": struct_kinds,
        "struct_target_ids": struct_target_ids_list,
        "struct_target_positions": struct_target_pos_list,
    }


print("Tokenizing...", flush=True)
tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names, num_proc=4)


class Collator:
    def __init__(self, tok, max_len):
        self.tok = tok
        self.max_len = max_len
        self.pad = tok.pad_token_id or 0
    def __call__(self, feats):
        ml = min(max(len(f["input_ids"]) for f in feats), self.max_len)
        ids, lbl, am = [], [], []
        struct_kinds, struct_target_ids, struct_target_pos = [], [], []
        for f in feats:
            x = f["input_ids"][:ml]
            y = f["labels"][:ml]
            a = f["attention_mask"][:ml]
            pl = ml - len(x)
            ids.append(x + [self.pad]*pl)
            lbl.append(y + [-100]*pl)
            am.append(a + [0]*pl)
            struct_kinds.append(f["struct_kind"])
            # 把 target_ids 转 tensor, target_positions 同样, 但要过滤超出 ml 的
            valid_pos = [p for p in f["struct_target_positions"] if p < ml]
            valid_ids = [t for t, p in zip(f["struct_target_ids"], f["struct_target_positions"]) if p < ml]
            struct_target_ids.append(torch.tensor(valid_ids, dtype=torch.long))
            struct_target_pos.append(torch.tensor(valid_pos, dtype=torch.long))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lbl, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(len(feats), ml, dtype=torch.long),
            "struct_kind": struct_kinds,  # list of strings
            "struct_target_ids": struct_target_ids,
            "struct_target_positions": struct_target_pos,
        }

collator = Collator(tokenizer, MAX_LENGTH)


class V5Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 把 list-typed inputs 单独处理
        struct_kind = inputs.pop("struct_kind", None)
        struct_target_ids = inputs.pop("struct_target_ids", None)
        struct_target_positions = inputs.pop("struct_target_positions", None)

        # 把 list-of-tensors 移到 device
        if struct_target_ids is not None:
            struct_target_ids = [t.to(model.base.device) for t in struct_target_ids]
        if struct_target_positions is not None:
            struct_target_positions = [t.to(model.base.device) for t in struct_target_positions]

        out = model(
            **inputs,
            struct_kind=struct_kind,
            struct_target_ids=struct_target_ids,
            struct_target_positions=struct_target_positions,
        )
        loss = out["loss"]
        return (loss, out) if return_outputs else loss

    def _save_lora(self, out_dir, model=None):
        m = model or self.model
        bm = m.base
        bm = bm.module if hasattr(bm, "module") else bm
        os.makedirs(out_dir, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in bm.state_dict().items() if "lora_" in k}
        torch.save(sd, os.path.join(out_dir, "lora_weights.pt"))
        ew = bm.get_input_embeddings().weight.detach().cpu()
        torch.save(ew, os.path.join(out_dir, "embedding_weights.pt"))
        # save classifier heads
        torch.save({
            "head_3di": m.head_3di.state_dict(),
            "head_dssp": m.head_dssp.state_dict(),
        }, os.path.join(out_dir, "struct_heads.pt"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(out_dir)

    def save_model(self, output_dir=None, _internal_call=False):
        self._save_lora(output_dir or self.args.output_dir)

    def _save_checkpoint(self, model, trial):
        cf = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        rd = self._get_output_dir(trial)
        od = os.path.join(rd, cf)
        self._save_lora(od, model=model)
        if self.args.should_save:
            self.state.save_to_json(os.path.join(od, TRAINER_STATE_NAME))
        self._save_optimizer_and_scheduler(od)
        self._save_rng_state(od)


training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=2,  # Structure 数据相对小, 多跑 epoch
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # 等效 batch=32 (Structure 数据少)
    optim="paged_adamw_8bit",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    max_grad_norm=1.0,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
)

trainer = V5Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=collator)

steps = len(tokenized) * 2 // (2 * 16)
print(f"\nBio-SFT v5 (with classification heads):")
print(f"  Init from: SFT v4")
print(f"  Train rows (Structure only): {len(tokenized)}")
print(f"  Joint loss: {ALPHA}*gen_CE + {BETA}*cls_CE")
print(f"  Total steps: ~{steps}")

trainer.train()

# 最终保存
print(f"\nSaving to {OUTPUT_DIR}...", flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items() if "lora_" in k}
torch.save(sd, os.path.join(OUTPUT_DIR, "lora_weights.pt"))
ew = base_model.get_input_embeddings().weight.detach().cpu()
torch.save(ew, os.path.join(OUTPUT_DIR, "embedding_weights.pt"))
torch.save({
    "head_3di": model.head_3di.state_dict(),
    "head_dssp": model.head_dssp.state_dict(),
}, os.path.join(OUTPUT_DIR, "struct_heads.pt"))
tokenizer.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "bio_sft_v5_meta.json"), "w") as f:
    json.dump({
        "init_from": SFT_V4_DIR,
        "task": "Structure (3Di/DSSP) with auxiliary classification heads",
        "alpha_gen": ALPHA,
        "beta_cls": BETA,
        "train_rows": len(tokenized),
    }, f, indent=2)
print("Done!")
