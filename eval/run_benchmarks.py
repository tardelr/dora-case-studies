import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import lm_eval
from lm_eval.models.huggingface import HFLM


# ── Config ──────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Load base model & tokenizer ─────────────────────────────────────

def load_base_model(model_id: str, quant_cfg: dict = None):
    print(f"Loading tokenizer & base model ({model_id}) …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if quant_cfg:
        bits = quant_cfg.get("bits", 4)
        if bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
    )
    return model, tokenizer


# ── Apply adapter (no merging) ───────────────────────────────────────

def apply_adapter(base_model, adapter_path: str):
    print(f"Loading adapter from {adapter_path} …")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("Adapter attached (unmerged, preserving quantization).")
    return model


# ── Run benchmarks ───────────────────────────────────────────────────

def run_benchmarks(model, tokenizer, tasks, num_fewshot=0, limit=None, batch_size=16):
    print(f"Running benchmarks: {tasks}")
    eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    return results


# ── Print results ────────────────────────────────────────────────────

def print_results(results: dict):
    print("\n── Results ──")
    for task, metrics in results["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc", "N/A"))
        acc_norm = metrics.get("acc_norm,none", metrics.get("acc_norm", "N/A"))
        print(f"  {task:>12s}  acc={acc}  acc_norm={acc_norm}")


# ── Export results & config ──────────────────────────────────────────

def export_results(results: dict, config_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, default=str, indent=2)
    print(f"Results saved to {results_path}")

    config_dest = os.path.join(output_dir, "eval_config.json")
    shutil.copy2(config_path, config_dest)
    print(f"Config saved to {config_dest}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lm-eval benchmarks — config-driven")
    parser.add_argument("--config", required=True, help="Path to eval config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)

    quant_cfg = cfg.get("quantization")
    base_model, tokenizer = load_base_model(cfg["base_model_id"], quant_cfg)

    adapter_path = cfg.get("adapter_path")
    if adapter_path:
        eval_model = apply_adapter(base_model, adapter_path)
    else:
        print("No adapter_path — evaluating base model directly.")
        eval_model = base_model

    results = run_benchmarks(
        eval_model,
        tokenizer,
        tasks=cfg["tasks"],
        num_fewshot=cfg.get("num_fewshot", 0),
        limit=cfg.get("limit"),
        batch_size=cfg.get("batch_size", 16),
    )
    print_results(results)

    output_dir = os.path.join("eval-results", cfg["model_name"])
    export_results(results, args.config, output_dir)
