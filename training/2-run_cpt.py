#!/usr/bin/env python
# coding: utf-8
"""
CPT Step 2: QLoRA 持续预训练
- 4-bit 量化 + LoRA (r=64) + 解冻 embedding + router
- 支持多卡 (accelerate / torchrun)
- 自定义 Dataset 从二进制文件流式读取
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Blackwell (sm_120) 不支持 torch._grouped_mm，强制走 loop fallback
import transformers.integrations.moe as _moe_module
_moe_module._can_use_grouped_mm = lambda *args, **kwargs: False

import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME
from peft import LoraConfig, inject_adapter_in_model

# ================= 1. 配置 =================
# v2: Instruct 扩容模型 + 新的数据目录
MODEL_DIR = "/root/autodl-tmp/dnagpt/models_local/gemma-4-26B-A4B-it-bio"
DATA_BIN = "/root/autodl-tmp/dnagpt/cpt_data_local/cpt_all.bin"
DATA_IDX = "/root/autodl-tmp/dnagpt/cpt_data_local/cpt_index.npy"
OUTPUT_DIR = "/root/autodl-tmp/dnagpt/outputs/gemma-4-26B-A4B-it-bio-cpt-v2"
CHECKPOINT_DIR = "/root/autodl-tmp/dnagpt/checkpoints_v2"

MAX_LENGTH = 1024

# ================= 2. 自定义 Dataset =================
class BinaryTokenDataset(Dataset):
    """从二进制文件 + 索引读取 tokenized chunks"""
    def __init__(self, bin_path, idx_path, max_length=1024):
        self.bin_path = bin_path
        self.index = np.load(idx_path)  # [[offset, length], ...]
        self.max_length = max_length
        self.file = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = open(self.bin_path, "rb")

        offset, length = self.index[idx]
        self.file.seek(int(offset) + 4)  # skip length header
        data = self.file.read(int(length) * 4)
        input_ids = np.frombuffer(data, dtype=np.uint32).astype(np.int64).tolist()

        # 截断到 max_length
        input_ids = input_ids[:self.max_length]

        return {
            "input_ids": input_ids,
            "labels": input_ids,
        }

# ================= 3. Data Collator =================
class CPTDataCollator:
    """动态 padding + mm_token_type_ids (Gemma4 要求)"""
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, features):
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            ids = f["input_ids"][:max_len]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            labels.append(f["labels"][:max_len] + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(len(features), max_len, dtype=torch.long),
        }
        return batch

# ================= 4. 加载模型 =================
print("Loading model (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map={"": int(os.environ.get("LOCAL_RANK", 0))},  # 每卡独立加载
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 5. 注入 LoRA + 解冻关键层 =================
print("Injecting LoRA...")

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 注入 LoRA 到 language_model
inject_adapter_in_model(lora_config, model.model.language_model, adapter_name="default")

# 解冻 embedding 层 (新 token 必须学习)
for param in model.get_input_embeddings().parameters():
    param.requires_grad = True
    # 保持 bfloat16，避免 Blackwell 上 MoE fallback 出现 float/bfloat16 dtype 冲突

# 让 transformers 识别已挂载 adapter
model._hf_peft_config_loaded = True

# 统计
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")

# ================= 6. 数据集 =================
print("Loading dataset...")
dataset = BinaryTokenDataset(DATA_BIN, DATA_IDX, MAX_LENGTH)
collator = CPTDataCollator(tokenizer, MAX_LENGTH)
print(f"  Total samples: {len(dataset):,}")

# ================= 7. 训练配置 =================
print("Setting up training...")
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=0.6,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,  # 等效 batch = 6*8卡*4 = 192
    optim="paged_adamw_8bit",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    max_grad_norm=1.0,
    logging_steps=50,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=3,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

class LoRATrainer(Trainer):
    def _save_lora_checkpoint(self, output_dir, model=None):
        model = model or self.model
        base_model = model.module if hasattr(model, "module") else model
        os.makedirs(output_dir, exist_ok=True)
        lora_state_dict = {k: v.detach().cpu() for k, v in base_model.state_dict().items() if "lora_" in k}
        torch.save(lora_state_dict, os.path.join(output_dir, "lora_weights.pt"))
        embed_weight = base_model.get_input_embeddings().weight.detach().cpu()
        torch.save(embed_weight, os.path.join(output_dir, "embedding_weights.pt"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        self._save_lora_checkpoint(output_dir)

    def _save_checkpoint(self, model, trial):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_lora_checkpoint(output_dir, model=model)
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        self._save_optimizer_and_scheduler(output_dir)
        self._save_rng_state(output_dir)
        if self.args.should_save:
            pass  # skip checkpoint rotation (method renamed in newer transformers)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model
        lora_path = os.path.join(resume_from_checkpoint, "lora_weights.pt")
        embed_path = os.path.join(resume_from_checkpoint, "embedding_weights.pt")
        if os.path.exists(lora_path):
            lora_state = torch.load(lora_path, map_location="cpu")
            ms = model.state_dict()
            for k, v in lora_state.items():
                if k in ms:
                    ms[k].copy_(v)
        if os.path.exists(embed_path):
            embed = torch.load(embed_path, map_location="cpu")
            model.get_input_embeddings().weight.data.copy_(embed)
        return model

# ================= 8. 训练 =================
trainer = LoRATrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,

)

print(f"\nStarting CPT training...")
print(f"  Epochs: 1")
print(f"  Batch size: {training_args.per_device_train_batch_size} × 8 GPUs × {training_args.gradient_accumulation_steps} accum = {training_args.per_device_train_batch_size * 8 * training_args.gradient_accumulation_steps}")
print(f"  Total steps: ~{len(dataset) // (training_args.per_device_train_batch_size * 8 * training_args.gradient_accumulation_steps)}")

trainer.train()

# ================= 9. 保存 =================
print(f"\nSaving to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存 LoRA 权重
lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
torch.save(lora_state_dict, os.path.join(OUTPUT_DIR, "lora_weights.pt"))

# 保存 embedding 权重 (包含新 token 的学习结果)
embed_weight = model.get_input_embeddings().weight.data
torch.save(embed_weight, os.path.join(OUTPUT_DIR, "embedding_weights.pt"))

# 保存 tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)

# 保存配置
import json
meta = {
    "base_model": MODEL_DIR,
    "lora_r": 64,
    "lora_alpha": 128,
    "target_modules": ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj', 'router.proj'],
    "trainable_params": trainable,
    "total_params": total,
    "total_tokens": sum(int(x[1]) for x in dataset.index),
}
with open(os.path.join(OUTPUT_DIR, "cpt_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Done! CPT model saved.")
