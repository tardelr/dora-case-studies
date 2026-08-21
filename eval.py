import argparse
import copy
import json
import math
import re
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import GenerationConfig

from train import (
    finalize_config,
    format_alpaca_text,
    load_model_and_tokenizer,
    read_yaml,
    save_yaml,
)

# ---------------------------------------------------------------------------
# Constants — copied from modern_pipeline/analysis_and_eval/paper_evaluation.py
# ---------------------------------------------------------------------------

ANSWER_PATTERNS = {
    "boolq": r"true|false",
    "piqa": r"solution1|solution2",
    "social_i_qa": r"answer1|answer2|answer3|answer4|answer5",
    "ARC-Challenge": r"answer1|answer2|answer3|answer4|answer5",
    "ARC-Easy": r"answer1|answer2|answer3|answer4|answer5",
    "openbookqa": r"answer1|answer2|answer3|answer4|answer5",
    "hellaswag": r"ending1|ending2|ending3|ending4",
    "winogrande": r"option1|option2",
}
DEFAULT_TASKS = list(ANSWER_PATTERNS.keys())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def extract_answer(task, generated_text):
    pattern = ANSWER_PATTERNS.get(task)
    if not pattern:
        raise ValueError(f"Unsupported eval task: {task}")
    matches = re.findall(pattern, generated_text.strip().lower())
    return matches[0] if matches else ""


def response_from_decoded(decoded_text):
    return decoded_text.split("### Response:")[-1].strip()


def format_inference_prompt(instruction, input_text=None):
    return format_alpaca_text(instruction, input_text, output="")


def make_run_dir(output_dir):
    results_root = Path(output_dir) / "paper_eval_results"
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"paper_eval_{timestamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions").mkdir()
    return run_dir


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def iter_batches(rows, batch_size):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


# ---------------------------------------------------------------------------
# Builders / loaders
# ---------------------------------------------------------------------------

def build_generation_config(eval_cfg):
    do_sample = bool(eval_cfg.get("do_sample", False))
    kwargs = {
        "do_sample": do_sample,
        "num_beams": eval_cfg.get("num_beams", 4),
        "max_length": None,
    }
    if do_sample:
        kwargs.update(
            {
                "temperature": eval_cfg.get("temperature", 0.1),
                "top_p": eval_cfg.get("top_p", 0.75),
                "top_k": eval_cfg.get("top_k", 40),
            }
        )
    return GenerationConfig(**kwargs)


def load_eval_task(task, eval_cfg):
    url = eval_cfg["task_url_template"].format(task=task)
    dataset = load_dataset("json", data_files=url, split="train")
    rows = list(dataset)
    limit = eval_cfg.get("limit")
    if limit:
        rows = rows[:limit]
    return rows


def load_adapter_if_available(model, config):
    experiment = config["experiment"]
    eval_cfg = config.get("eval", {})
    adapter_path = Path(
        eval_cfg.get("adapter_path") or Path(experiment["output_dir"]) / "final_adapter"
    )
    if adapter_path.exists():
        print(f"Loading adapter from {adapter_path}")
        return PeftModel.from_pretrained(model, adapter_path)
    if eval_cfg.get("allow_base_model_eval", False):
        print(f"No adapter found at {adapter_path}; evaluating base model.")
        return model
    raise FileNotFoundError(f"No adapter found at {adapter_path}")


def prepare_tokenizer_for_generation(tokenizer):
    original_padding_side = tokenizer.padding_side
    # Decoder-only generation with batching requires left padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return original_padding_side


def generate_responses(model, tokenizer, prompts, eval_cfg, gen_config):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            generation_config=gen_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=eval_cfg.get("max_new_tokens", 32),
            return_dict_in_generate=True,
            output_scores=False,
        )

    decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
    return [response_from_decoded(text) for text in decoded]


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(model, tokenizer, task, eval_cfg, gen_config, run_dir):
    rows = load_eval_task(task, eval_cfg)
    predictions = []
    correct = 0
    parse_failures = 0
    batch_size = eval_cfg.get("batch_size", 16)
    save_every = max(int(eval_cfg.get("save_every_n_batches", 1)), 1)
    prediction_path = run_dir / "predictions" / f"{task}.json"
    total_batches = math.ceil(len(rows) / batch_size) if rows else 0
    print(
        f"\nEval {task}: {len(rows)} examples, "
        f"batch_size={batch_size}, batches={total_batches}"
    )

    batch_iterator = tqdm(
        iter_batches(rows, batch_size),
        total=total_batches,
        desc=f"eval {task}",
        unit="batch",
        dynamic_ncols=True,
    )
    for batch_idx, batch in enumerate(batch_iterator, start=1):
        prompts = [
            format_inference_prompt(row.get("instruction", ""), row.get("input") or None)
            for row in batch
        ]
        outputs = generate_responses(model, tokenizer, prompts, eval_cfg, gen_config)

        for row, prompt, output in zip(batch, prompts, outputs):
            gold = str(row.get("answer", "")).strip().lower()
            pred = extract_answer(task, output)
            is_correct = gold == pred
            correct += int(is_correct)
            parse_failures += int(pred == "")

            result_row = copy.deepcopy(row)
            result_row.update(
                {
                    "task": task,
                    "prompt": prompt,
                    "gold": gold,
                    "output_pred": output,
                    "pred": pred,
                    "correct": is_correct,
                    "flag": is_correct,
                }
            )
            predictions.append(result_row)

        processed = len(predictions)
        accuracy_so_far = correct / processed if processed else 0.0
        batch_iterator.set_postfix(
            acc=f"{accuracy_so_far:.4f}",
            correct=f"{correct}/{processed}",
            parse_failures=parse_failures,
        )

        if batch_idx % save_every == 0:
            write_json(prediction_path, predictions)

    write_json(prediction_path, predictions)
    total = len(rows)
    accuracy = correct / total if total else 0.0
    print(
        f"Finished {task}: acc={accuracy:.4f}, "
        f"correct={correct}/{total}, parse_failures={parse_failures}"
    )
    return {
        "task": task,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "parse_failures": parse_failures,
        "prediction_file": str(prediction_path),
    }


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def save_eval_results(run_dir, config, task_results):
    accuracies = [result["accuracy"] for result in task_results.values()]
    results = {
        "protocol": "dora_llm_adapters_generation",
        "model": config.get("model", {}),
        "experiment": config.get("experiment", {}),
        "adapter": config.get("adapter", {}),
        "quantization": config.get("quantization", {}),
        "eval": config.get("eval", {}),
        "results": task_results,
        "average_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
    }
    results_path = run_dir / "results.json"
    write_json(results_path, results)
    save_yaml(config, run_dir / "eval_resolved_config.yaml")
    print(f"Saved eval results to {results_path}")
    return results_path


def print_results(task_results):
    print("\nResults")
    for task, result in task_results.items():
        print(
            f"  {task:>14s}  acc={result['accuracy']:.4f}  "
            f"correct={result['correct']}/{result['total']}  "
            f"parse_failures={result['parse_failures']}"
        )
    accuracies = [result["accuracy"] for result in task_results.values()]
    average = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f"  {'average':>14s}  acc={average:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper-faithful generative eval for simple/ runs.")
    parser.add_argument("--config", default="./train_config.yaml")
    parser.add_argument("--adapter-path", default=None, help="Override eval.adapter_path.")
    parser.add_argument("--limit", type=int, default=None, help="Override eval.limit for smoke runs.")
    args = parser.parse_args()

    config = finalize_config(read_yaml(args.config))
    eval_cfg = config.setdefault("eval", {})
    if args.adapter_path:
        eval_cfg["adapter_path"] = args.adapter_path
    if args.limit is not None:
        eval_cfg["limit"] = args.limit

    model, tokenizer = load_model_and_tokenizer(config)
    model = load_adapter_if_available(model, config)
    model.eval()

    original_padding_side = prepare_tokenizer_for_generation(tokenizer)
    gen_config = build_generation_config(eval_cfg)
    run_dir = make_run_dir(config["experiment"]["output_dir"])
    try:
        task_results = {}
        for task in (eval_cfg.get("tasks") or DEFAULT_TASKS):
            task_results[task] = evaluate_task(
                model, tokenizer, task, eval_cfg, gen_config, run_dir
            )
            save_eval_results(run_dir, config, task_results)
    finally:
        tokenizer.padding_side = original_padding_side

    print_results(task_results)


if __name__ == "__main__":
    main()
