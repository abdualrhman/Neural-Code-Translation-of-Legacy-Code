import json
import numpy as np
import torch
import re
from typing import List
from datasets import load_dataset, Dataset, Value
import logging
import os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer")


CONFIG = {
    "model_name": "Qwen/Qwen3-32B",
    "output_dir": "/work3/s204476/Qwen3-32B_3",
    "results_dir": "../results",
    "max_samples": None,
    "batch_size": 8,
    "max_length": 2048,
    "gen_max_new_tokens": 2048,
    "num_beams": 1,
    "early_stopping": True,
    "use_cache": True,
    "use_flash_attention": False,
    "tokenizer_skip_special_tokens": True,
    "test_data_file": "../data/Bx_test_val.json",
    "data_dir": "../data/",
    "seed": 42,
}


def extract_csharp(text: str) -> str:
    md_pattern = re.compile(
        r"```(?:csharp|C#)?\s*(.*?)```",
        re.DOTALL
    )
    md_blocks = md_pattern.findall(text)

    if md_blocks:
        last_block = md_blocks[-1].strip()
        return last_block

    code_start_keywords = [
        "using ",
        "namespace ",
        "class ",
        "public ",
        "internal ",
        "static "
    ]

    starts = []
    for kw in code_start_keywords:
        idx = text.rfind(kw)
        if idx != -1:
            starts.append(idx)

    if not starts:
        return text.strip()

    start = max(starts)
    candidate = text[start:]

    brace_count = 0
    end_index = None

    for i, ch in enumerate(candidate):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end_index = i
    if end_index is not None:
        cleaned = candidate[:end_index + 1]
        return cleaned.strip()

    return candidate.strip()


def build_prompt(
    tokenizer,
    apl,
    apl_desc,
    principal_method_name,
    method_signatures,
):
    system_prompt = """You are an expert APL code programmer.\n
            Given the following APL code create C# program that implements the given code.\n"""
    user_content = f"### APL code:\n{apl}\n"
    if method_signatures:
        user_content += (
            "\n### Required C# method signatures (overloads)\n"
            "You MUST implement all of the following overloads exactly as written "
            "(names, return types, and parameter types):\n"
            f"{method_signatures}\n"
        )
    user_content += "Output format: Only compilable C# program code, no explanations, no reasoning, no example usage.\n"
    user_content += "### C#:\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False
    )


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(cfg):
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if cfg.get("use_flash_attention") else None,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    adapter_dir = cfg.get("output_dir", None)
    if adapter_dir and os.path.exists(adapter_dir):
        try:
            logger.info(f"Trying to load LoRA adapters from {adapter_dir} …")
            model = PeftModel.from_pretrained(base, adapter_dir)
            logger.info("Successfully loaded pretrained adapters.")
        except Exception as e:
            logger.warning(f"Could not load pretrained adapters: {e}")
            logger.info("Falling back to base model (fresh, no adapters).")
            model = base
    else:
        logger.info("No pretrained adapters found. Using fresh base model.")
        model = base

    model.eval()
    return model, tokenizer


def load_test_dataset(cfg) -> Dataset:
    if cfg["test_data_file"] and os.path.exists(cfg["test_data_file"]):
        logger.info(f"Loading test set from {cfg['test_data_file']}")
        df = pd.read_json(
            cfg["test_data_file"],
            lines=True if cfg["test_data_file"].endswith("l") else False
        )
        df = df.drop(columns=['io'])
        ds = Dataset.from_pandas(df)
        missing = [k for k in ["apl", "csharp"] if k not in ds.column_names]
        if missing:
            raise ValueError(f"Test file missing required fields: {missing}")
        return ds


@torch.no_grad()
def batched_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    num_beams: int,
    early_stopping: bool,
    batch_size: int,
):
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            padding_side='left',
            truncation=True,
            max_length=CONFIG["max_length"],
        ).to(model.device)

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
            do_sample=CONFIG["num_beams"] == 1,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=CONFIG.get("use_cache", True)
        )
        gen_only = gen[:, enc["input_ids"].shape[1]:]

        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        outputs.extend(decoded)

        logger.info(f"Generated {min(start + batch_size, len(prompts))}/{len(prompts)}")

    return outputs


def save_results(preds):
    out_file = CONFIG['results_dir'] + "/Qwen3_Bx_test_val_f.json"
    test_data = []
    if CONFIG["test_data_file"].endswith("l"):
        with open(CONFIG["test_data_file"], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
    else:
        with open(CONFIG["test_data_file"], 'r', encoding='utf-8') as f:
            test_data = json.load(f)

    test_data = [{**item, "model_pred": preds[i]} for i, item in enumerate(test_data[:len(preds)])]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Predictions saved to {out_file}")


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    test_ds = load_test_dataset(cfg)

    n_total = len(test_ds)
    if cfg["max_samples"] is not None:
        n_eval = min(cfg["max_samples"], n_total)
        logger.info(f"Sampling first {n_eval} examples for eval.")
        test_ds = test_ds.select(range(n_eval))
    else:
        n_eval = n_total

    model, tokenizer = load_model_and_tokenizer(cfg)

    prompts = []
    for datapoint in test_ds:
        apl = datapoint["apl"]
        apl_desc = datapoint["description"]
        principal_method_name = datapoint["method_name"]
        method_signatures = datapoint["method_signatures"]
        prompts.append(build_prompt(tokenizer, apl, apl_desc, principal_method_name, method_signatures))

    raw_outputs = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=cfg["gen_max_new_tokens"],
        num_beams=cfg["num_beams"],
        early_stopping=cfg["early_stopping"],
        batch_size=cfg["batch_size"],
    )
    preds = [o for o in raw_outputs]
    save_results(preds)


if __name__ == "__main__":
    main()
