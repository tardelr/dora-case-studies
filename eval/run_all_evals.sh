#!/usr/bin/env bash
# Run lm-eval benchmarks for base, qlora, and qdora models.
# Edit the adapter paths and model names below as needed.
set -euo pipefail
cd "$(dirname "$0")/.."

# Base model (no adapter)
python eval/run_benchmarks.py --config eval/evals_config.json --model-name base

# QLoRA
python eval/run_benchmarks.py --config eval/evals_config.json --model-name qlora \
    --adapter-path ./qwen2.5-1.5b-lora-commonsense/final_adapter

# QDoRA
python eval/run_benchmarks.py --config eval/evals_config.json --model-name qdora \
    --adapter-path ./qwen2.5-1.5b-dora-commonsense/final_adapter
