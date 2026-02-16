from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import logging
import random
import wandb
from transformers.integrations import WandbCallback
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CONFIG = {
    "model_name": "Qwen/Qwen3-32B",
    "output_dir": "/work3/s204476/Qwen3-32B-nl-to-cs",
    "num_epochs": 4,
    "train_batch_size": 2,
    "eval_batch_size": 2,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "gradient_accumulation_steps": 2,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "learning_rate": 2e-5,
    "logging_steps": 50,
    "save_steps": 500,
    "eval_on_start": True,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "max_grad_norm": 0.3,
    "tokenizer_skip_special_tokens": True,
    "max_length": 512,
    "data_dir": "../data/",
    "train_data_file": "../data/B_nl_train.json",
    "val_data_file": "../data/ABC_nl_val.json",
    "seed": 42,
    "epoch_log_interval": 5,
    "eval_sample_size": 512,
    "eval_gen_max_new_tokens": 256,
    "training_info_path": "training_info.json",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CODE_TAG = "### C#:\n"


def build_chat_prompt(tokenizer, apl, nl_description):
    system_content = """
        You are an expert APL code programmer.
        Given the following APL code and natural language summary of a program, create C# program that implements the given summary. 
        """
    user_content = f"### APL code: {apl}\n"
    user_content = f"### Natural language summary: {nl_description}\n"
    user_content += "Output format: Only compilable C# program code, no explanations, no reasoning, no example usage."
    user_content += "### C#:"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False
    )
    return prompt


def build_full_text(tokenizer, apl, nl_description, csharp):
    prompt = build_chat_prompt(tokenizer, apl, nl_description)
    full_text = prompt + csharp + tokenizer.eos_token
    return full_text, len(tokenizer.encode(prompt, add_special_tokens=False))


@dataclass
class CustomDataCollatorForCausalLM:
    tokenizer: Any
    max_length: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        max_len = max(len(ids) for ids in input_ids)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        pad_token_id = self.tokenizer.pad_token_id
        
        for ids, labs in zip(input_ids, labels):
            if len(ids) > max_len:
                ids = ids[:max_len]
                labs = labs[:max_len]
            
            padding_length = max_len - len(ids)
            
            padded_ids = [pad_token_id] * padding_length + ids
            padded_labs = [-100] * padding_length + labs
            attn_mask = [0] * padding_length + [1] * len(ids)
            
            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_labs)
            attention_mask.append(attn_mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def tokenize_and_mask_chat(batch, tokenizer, max_length):
    input_ids_list = []
    labels_list = []
    
    for i in range(len(batch["apl"])):
        apl = batch["apl"][i]
        nl_description = batch["nl_description"][i]
        csharp = batch["csharp"][i]
        
        full_text, prompt_token_len = build_full_text(tokenizer, apl, nl_description, csharp)
        
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        
        input_ids = encoded["input_ids"]
        labels = input_ids[:]
        
        for j in range(min(prompt_token_len, len(labels))):
            labels[j] = -100
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }


def load_and_split_dataset(cfg) -> DatasetDict:
    def _load_split(path: str, split_name: str):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"{split_name} file not found or unset: {path}")

        logger.info(f"Loading {split_name} set from {path}")
        df = pd.read_json(path)
        
        if "io" in df.columns:
            df = df.drop(columns=["io"])

        required = ['apl', 'csharp', 'nl_description']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{split_name} file missing required columns: {missing}. "
                f"Columns present: {list(df.columns)}"
            )

        df['apl'] = df['apl'].astype(str).fillna('')
        df['nl_description'] = df['nl_description'].astype(str).fillna('')
        
        if 'description' not in df.columns:
            df['description'] = ''
        else:
            df['description'] = df['description'].astype(str).fillna('')
        
        if 'method_name' not in df.columns:
            df['method_name'] = None
        
        if 'method_signatures' not in df.columns:
            df['method_signatures'] = None

        ds = Dataset.from_pandas(df, preserve_index=False)
        ds = ds.shuffle(seed=cfg["seed"])
        logger.info(f"{split_name} size: {len(ds)}")
        return ds

    train_ds = _load_split(cfg.get("train_data_file", ""), "train")
    val_ds = _load_split(cfg.get("val_data_file", ""), "validation")

    ds = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
    })
    logger.info(
        f"Dataset sizes -> train: {len(ds['train'])}, "
        f"val: {len(ds['validation'])}"
    )
    return ds


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    ds = load_and_split_dataset(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = ds.map(
        lambda x: tokenize_and_mask_chat(x, tokenizer, cfg["max_length"]),
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing dataset"
    )

    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_cfg,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_cfg = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    data_collator = CustomDataCollatorForCausalLM(
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
    )
    steps_per_epoch = len(ds["train"]) // (
        cfg['train_batch_size'] * cfg['gradient_accumulation_steps']
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_grad_norm=cfg["max_grad_norm"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        learning_rate=cfg["learning_rate"],
        logging_steps=2 * steps_per_epoch,
        save_steps=cfg["save_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_on_start=cfg["eval_on_start"],
        save_strategy="epoch",
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        report_to=["wandb"],
        run_name="qwen3-nl-to-cs-lora",
        logging_first_step=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

    wandb.init(
        project="nl-to-cs",
        group="Qwen3-qLoRA",
        name=training_args.run_name,
        config=CONFIG,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )
    trainer.add_callback(WandbCallback())

    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
