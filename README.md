# DoRA Case Studies

This project is built to reproduce the original DoRA paper results[1], and assess if the authors' results would hold under quantization. _Publication pending_

The method itself is very similar to what would be necessary to train an arbitrary LoRA adapter, but there are specificities used to handle special DoRA cases. 

The notebooks with results' analysis were made available, but the underlying source files isn't because it'd require too much storage.

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

_(to be added)_


## References
[1] _Add reference to the DoRA paper_