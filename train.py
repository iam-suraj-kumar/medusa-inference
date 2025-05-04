import logging
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments)

from .medusa import add_medusa_heads
from .sft import MedusaSFTTrainer
from .utils import ScriptArguments, freeze_layers, save_medusa_heads

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()
training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

################
# Dataset
################
train_dataset = load_dataset(
    "json",
    data_files=script_args.train_dataset_path,
    split="train",
)
eval_dataset = load_dataset(
    "json",
    data_files=script_args.eval_dataset_path,
    split="train",
)

################
# Model & Tokenizer
################
torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_path,
    device_map="auto",
    torch_dtype=torch_dtype,
    quantization_config=quantization_config,
)
# Add medusa heads and freeze base model
add_medusa_heads(
    model,
    medusa_num_layers=1,
    medusa_num_heads=script_args.medusa_num_heads,
)
freeze_layers(model)
model.config.torch_dtype = torch_dtype
model.config.use_cache = False
logger.info("Finished loading model and medusa heads")

tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

################
# Training
################
trainer = MedusaSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
    },
    medusa_num_heads=script_args.medusa_num_heads,
    medusa_heads_coefficient=script_args.medusa_heads_coefficient,
    medusa_decay_coefficient=script_args.medusa_decay_coefficient,
    medusa_scheduler=script_args.medusa_scheduler,
    train_only_medusa_heads=script_args.train_only_medusa_heads,
    medusa_lr_multiplier=script_args.medusa_lr_multiplier,
)
trainer.train()

##########################
# Save tokenizer, medusa heads
##########################
save_dir = "./model/medusa-model"
trainer.tokenizer.save_pretrained(save_dir)
save_medusa_heads(
    save_dir, script_args.medusa_num_heads, model, torch_dtype
)
