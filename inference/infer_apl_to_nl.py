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
    "output_dir": "/work3/s204476/Qwen3-32B-apl-to-nl_____",
    "results_dir": "../results",
    "max_samples": None,
    "batch_size": 8,
    "max_length": 1024,
    "gen_max_new_tokens": 1024,
    "num_beams": 1,
    "early_stopping": True,
    "use_cache": True,
    "tokenizer_skip_special_tokens": True,
    "test_data_file": "../data/Bx_test_val.json",
    "data_dir": "../data/",
    "seed": 42,
}


def extract_qwen_output(text):
    output = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return output.strip()


def build_prompt(
    tokenizer,
    apl_code,
):
    system_content = """ 
You are an APL programming language expert.
Create a natural language summary of the following APL code. The summary will be used to re implement the program in another language.
Focus on providing a summary that include a precise description of code's functionality and it should not be ambiguous.
include in the summary the input and output types. 
Output only the summary.
        """
    user_content = f"### APL code: {apl_code}\n"
    messages = [
        {"role": "system", "content": system_content},
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
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
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
    out_file = CONFIG['results_dir'] + "/qwen3_apl_to_nl_Bx_test_val.json"
    test_data = []
    if CONFIG["test_data_file"].endswith("l"):
        with open(CONFIG["test_data_file"], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
    else:
        with open(CONFIG["test_data_file"], 'r', encoding='utf-8') as f:
            test_data = json.load(f)

    test_data = [{**item, "apl_to_nl_pred": preds[i]} for i, item in enumerate(test_data[:len(preds)])]
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
        apl_code = datapoint["apl"]
        prompts.append(build_prompt(tokenizer, apl_code))

    raw_outputs = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=cfg["gen_max_new_tokens"],
        num_beams=cfg["num_beams"],
        early_stopping=cfg["early_stopping"],
        batch_size=cfg["batch_size"],
    )
    outputs = [extract_qwen_output(o) for o in raw_outputs]
    save_results(outputs)


if __name__ == "__main__":
    main()
