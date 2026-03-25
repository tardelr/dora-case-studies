import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import lm_eval
from lm_eval.models.huggingface import HFLM


# ── Config ──────────────────────────────────────────────────────────

def load_config(path: str = "evals_config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── Load base model & tokenizer ─────────────────────────────────────

def load_base_model(model_id: str):
    print(f"Loading tokenizer & base model ({model_id}) …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer


# ── Merge adapter ────────────────────────────────────────────────────

def merge_adapter(base_model, adapter_path: str):
    print(f"Loading & merging adapter from {adapter_path} …")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()
    print("Adapter merged successfully.")
    return merged


# ── Run benchmarks ───────────────────────────────────────────────────

def run_benchmarks(model, tokenizer, tasks: list[str], num_fewshot: int = 0, limit=None, batch_size: int = 16):
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


# ── Export results ───────────────────────────────────────────────────

def export_results(results: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, default=str, indent=2)
    print(f"Results saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lm-eval benchmarks — config-driven")
    parser.add_argument("--config", default="eval/evals_config.json", help="Path to evals config JSON")
    parser.add_argument("--model-name", required=True,
                        help="Name for this model run (used as subfolder in eval-results/)")
    parser.add_argument("--adapter-path", default=None,
                        help="Override adapter_path from config (omit to eval base model)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_model, tokenizer = load_base_model(cfg["base_model_id"])

    adapter_path = args.adapter_path or cfg.get("adapter_path")
    if adapter_path:
        eval_model = merge_adapter(base_model, adapter_path)
    else:
        print("No adapter_path set — evaluating base model directly.")
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

    output_path = os.path.join("eval-results", args.model_name, "results.json")
    export_results(results, output_path)
