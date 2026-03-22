import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import lm_eval
from lm_eval.models.huggingface import HFLM


# ── Config ──────────────────────────────────────────────────────────

BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR = "./meta-llama-8b-dora-commonsense"
ADAPTER_PATH = f"{OUTPUT_DIR}/final_adapter"
TASKS = ["hellaswag", "arc_easy"]
NUM_FEWSHOT = 0
LIMIT = None  # set to e.g. 50 for a quick sanity check


# ── Load base model & tokenizer ─────────────────────────────────────

def load_base_model(model_id: str):
    print("Loading tokenizer & base model …")
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
    with open(output_path, "w") as f:
        json.dump(results, f, default=str)
    print(f"Results saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lm-eval benchmarks on a LoRA/DoRA fine-tuned model")
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--num-fewshot", type=int, default=NUM_FEWSHOT)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="results_llama_8b_dora.json")
    args = parser.parse_args()

    base_model, tokenizer = load_base_model(args.base_model)
    merged_model = merge_adapter(base_model, args.adapter_path)
    results = run_benchmarks(merged_model, tokenizer, args.tasks, args.num_fewshot, args.limit, args.batch_size)
    print_results(results)
    export_results(results, args.output)
