#!/usr/bin/env bash
# Run lm-eval benchmarks for all model configs.
set -euo pipefail
cd "$(dirname "$0")/.."

python eval/run_benchmarks.py --config eval/config_base.json
python eval/run_benchmarks.py --config eval/config_qlora.json
python eval/run_benchmarks.py --config eval/config_qdora.json
