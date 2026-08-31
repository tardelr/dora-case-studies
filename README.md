# DoRA Case Studies

This project is built to reproduce the original DoRA paper results[1], and assess if the authors' results would hold under quantization. _Publication pending_

The method itself is very similar to what would be necessary to train an arbitrary LoRA adapter, but there are specificities used to handle special DoRA cases. 

The notebooks with results' analysis were made available, but the underlying source files aren't because they'd require too much storage. The trained adapters are published on Hugging Face: [tardelr/lora-dora-reproducibility](https://huggingface.co/tardelr/lora-dora-reproducibility).

## Training

```bash
python train.py --config path/to/config.yaml
```

Single-file LoRA/DoRA/QLoRA/QDoRA trainer. Output is written to `{experiment.output_dir}` (defaults to `simple/outputs/{experiment.name}/`), containing:

- `resolved_config.yaml` — fully resolved config used for the run
- `run_metadata.json` — model/param/hardware info
- `metrics/` — `trainer_log.jsonl`, `resource_stats.jsonl`, `gradient_stats.jsonl`, `magnitude_grad_norms.jsonl`, `magnitude_audit.json`
- `checkpoints/` — periodic Trainer checkpoints (for resuming)
- `final_adapter/` — final saved adapter + tokenizer

## Evaluation

```bash
python eval.py --config path/to/config.yaml [--adapter-path ...] [--limit N]
```

Uses the **same config** as training (loads the adapter from `final_adapter/` under `experiment.output_dir` by default). Runs the paper-style generative eval across the tasks in `eval.tasks`.

Output is written to `{experiment.output_dir}/paper_eval_results/paper_eval_{timestamp}/`, containing:

- `predictions/{task}.json` — per-example prompts, gold/predicted answers, correctness
- `results.json` — per-task accuracy + overall average
- `eval_resolved_config.yaml` — config used for the eval run

## Notebooks

Analysis notebooks live in `notebooks/`. Each one takes a hardcoded list of run directories (edit the paths at the top before running) and writes tables/plots to `notebooks/outputs/`.

- `training_stats.ipynb` — sanity check on a single run's training logs (loss, token accuracy, gradient norms, magnitude audit) before spending compute on evaluation.
- `audit_runs.ipynb` — reads `resolved_config.yaml` / `run_metadata.json` across all runs and builds comparative tables of hyperparameters, adapter settings, quantization and hardware.
- `eval_analysis.ipynb` — aggregates `paper_eval` results into one row per run and one column per benchmark, plus per-task correct/total counts and answer parse-failure rates.
- `mechanistic_analysis.ipynb` — reproduces the paper's weight-decomposition analysis: per-layer magnitude (ΔM) and direction (ΔD) changes across checkpoints, their regression slopes, and low-rank coefficient-update growth.

## Trained models

All adapters (LoRA, DoRA, QLoRA, QDoRA, across seeds) are available at [tardelr/lora-dora-reproducibility](https://huggingface.co/tardelr/lora-dora-reproducibility).
