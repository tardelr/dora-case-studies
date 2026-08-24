import argparse
import copy
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPACA_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def resolve_dtype(dtype_name):
    if dtype_name in (None, "auto"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    normalized = str(dtype_name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def sanitize_for_path(value):
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-")


def infer_method(config):
    adapter = config.get("adapter", {})
    quantization = config.get("quantization", {})
    use_dora = bool(adapter.get("use_dora", False))
    quantized = bool(quantization.get("enabled", False))
    if use_dora and quantized:
        return "qdora"
    if use_dora:
        return "dora"
    if quantized:
        return "qlora"
    return "lora"


def format_alpaca_text(instruction, input_text=None, output=""):
    if input_text:
        return ALPACA_WITH_INPUT.format(instruction=instruction, input=input_text, output=output)
    return ALPACA_WITHOUT_INPUT.format(instruction=instruction, output=output)


def format_example(example, eos_token):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    example["text"] = format_alpaca_text(instruction, input_text, output) + eos_token
    return example


# ---------------------------------------------------------------------------
# Callback classes (HF Trainer API requires subclasses of TrainerCallback)
# ---------------------------------------------------------------------------

class JsonlLoggingCallback(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {"step": state.global_step, "epoch": state.epoch, **logs}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")


class ResourceStatsCallback(TrainerCallback):
    def __init__(self, path, every_n_logs=1):
        self.path = Path(path)
        self.every_n_logs = max(1, every_n_logs)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = None
        self.log_count = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_idx)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_count += 1
        if self.log_count % self.every_n_logs != 0:
            return
        self._write_stats(state, logs or {}, event="log")

    def on_train_end(self, args, state, control, **kwargs):
        self._write_stats(state, {}, event="train_end")

    def _write_stats(self, state, logs, event):
        payload = {
            "event": event,
            "step": state.global_step,
            "epoch": state.epoch,
            "elapsed_seconds": (
                time.perf_counter() - self.start_time if self.start_time is not None else None
            ),
        }
        for key in (
            "loss",
            "learning_rate",
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "num_tokens",
        ):
            if key in logs:
                payload[key] = logs[key]

        if torch.cuda.is_available():
            devices = []
            for device_idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device_idx)
                devices.append(
                    {
                        "index": device_idx,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / 1024**3,
                        "memory_allocated_gb": torch.cuda.memory_allocated(device_idx) / 1024**3,
                        "memory_reserved_gb": torch.cuda.memory_reserved(device_idx) / 1024**3,
                        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device_idx) / 1024**3,
                        "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device_idx) / 1024**3,
                    }
                )
            payload["cuda_devices"] = devices
        else:
            payload["cuda_devices"] = []

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")


class GradientStatsCallback(TrainerCallback):
    def __init__(self, path, every_n_steps=50):
        self.path = Path(path)
        self.every_n_steps = max(every_n_steps, 1)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Separate log file for per-magnitude-param gradient norms.
        # Written alongside gradient_stats.jsonl as magnitude_grad_norms.jsonl.
        self.magnitude_path = self.path.parent / "magnitude_grad_norms.jsonl"

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step % self.every_n_steps != 0:
            return

        grad_sum = 0.0
        grad_sq_sum = 0.0
        grad_abs_sum = 0.0
        grad_count = 0
        trainable_param_count = 0
        tensors_with_grad = 0

        # Per-magnitude-param records collected in the same pass.
        magnitude_records = []

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable_param_count += parameter.numel()

            # --- magnitude-specific tracking ---
            if _is_magnitude_param(name):
                has_grad = parameter.grad is not None
                magnitude_records.append({
                    "param_name": name,
                    "has_grad": has_grad,
                    "grad_l2_norm": (
                        parameter.grad.detach().float().norm().item() if has_grad else None
                    ),
                    "grad_abs_mean": (
                        parameter.grad.detach().float().abs().mean().item() if has_grad else None
                    ),
                    "param_norm": parameter.detach().float().norm().item(),
                    "dtype": str(parameter.dtype),
                })

            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            tensors_with_grad += 1
            grad_count += grad.numel()
            grad_sum += grad.sum().item()
            grad_sq_sum += grad.pow(2).sum().item()
            grad_abs_sum += grad.abs().sum().item()

        if grad_count == 0:
            return

        mean = grad_sum / grad_count
        variance = max((grad_sq_sum / grad_count) - (mean * mean), 0.0)
        payload = {
            "step": state.global_step,
            "epoch": state.epoch,
            "trainable_param_count": trainable_param_count,
            "tensors_with_grad": tensors_with_grad,
            "grad_count": grad_count,
            "grad_mean": mean,
            "grad_variance": variance,
            "grad_std": math.sqrt(variance),
            "grad_l2_norm": math.sqrt(grad_sq_sum),
            "grad_abs_mean": grad_abs_sum / grad_count,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

        # Write magnitude records only when there are magnitude params present
        # (i.e. DoRA / QDoRA runs).  This keeps the file absent for LoRA runs.
        if magnitude_records:
            magnitude_payload = {
                "step": state.global_step,
                "epoch": state.epoch,
                "magnitude_params": magnitude_records,
                # Convenience roll-up: are ALL magnitude params receiving grads?
                "all_have_grad": all(r["has_grad"] for r in magnitude_records),
                "any_have_grad": any(r["has_grad"] for r in magnitude_records),
            }
            with self.magnitude_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(magnitude_payload, default=str) + "\n")


class AdapterSnapshotCallback(TrainerCallback):
    def __init__(self, output_dir, fractions=None, save_final=True):
        self.output_dir = Path(output_dir)
        self.fractions = sorted(fractions or [0.25, 0.5, 0.75])
        self.save_final = save_final
        self.saved_labels = set()
        self.snapshot_root = self.output_dir / "analysis_snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or not state.max_steps:
            return
        for fraction in self.fractions:
            label = f"fraction_{fraction:g}"
            threshold = max(1, round(state.max_steps * fraction))
            if label not in self.saved_labels and state.global_step >= threshold:
                self._save_snapshot(model, state, label)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None and self.save_final:
            self._save_snapshot(model, state, "final")

    def _save_snapshot(self, model, state, label):
        if label in self.saved_labels:
            return
        snapshot_dir = self.snapshot_root / f"{label}_step_{state.global_step}"
        model.save_pretrained(snapshot_dir)
        metadata = {
            "label": label,
            "step": state.global_step,
            "epoch": state.epoch,
            "max_steps": state.max_steps,
        }
        with (snapshot_dir / "snapshot_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        self.saved_labels.add(label)



# ---------------------------------------------------------------------------
# DoRA magnitude-vector diagnostic: requires_grad + dtype audit
# ---------------------------------------------------------------------------

# Parameter name fragments that identify DoRA magnitude vectors across PEFT versions.
_MAGNITUDE_KEYWORDS = ("lora_magnitude_vector", "lora_magnitude", "magnitude_vector")


def _is_magnitude_param(name: str) -> bool:
    return any(kw in name for kw in _MAGNITUDE_KEYWORDS)


def run_magnitude_audit(model, output_path: Path) -> None:
    """
    Inspect every DoRA magnitude parameter and log its requires_grad, dtype,
    shape, and device to a JSON file.  Call this once after the trainer is
    built so the PEFT adapter has already been attached and prepare_model_for_
    kbit_training has run.

    The resulting file (magnitude_audit.json) is the first thing to check when
    investigating a frozen magnitude vector: if requires_grad is False for any
    entry, that parameter will never receive a gradient update.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for name, param in model.named_parameters():
        if not _is_magnitude_param(name):
            continue
        entries.append({
            "param_name": name,
            "requires_grad": param.requires_grad,
            "dtype": str(param.dtype),
            "shape": list(param.shape),
            "device": str(param.device),
            "numel": param.numel(),
        })

    # Also count how many magnitude params are trainable vs frozen.
    n_trainable = sum(1 for e in entries if e["requires_grad"])
    n_frozen    = len(entries) - n_trainable
    summary = {
        "total_magnitude_params": len(entries),
        "trainable": n_trainable,
        "frozen": n_frozen,
        "frozen_names": [e["param_name"] for e in entries if not e["requires_grad"]],
    }

    payload = {"summary": summary, "params": entries}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Print a short human-readable summary so it also appears in the run log.
    status = "OK – all trainable" if n_frozen == 0 else f"WARNING – {n_frozen} frozen"
    print(
        f"[magnitude_audit] {len(entries)} magnitude params found | "
        f"trainable={n_trainable} | frozen={n_frozen} | {status}"
    )
    if n_frozen:
        for name in summary["frozen_names"]:
            print(f"  FROZEN: {name}")

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_quantization_config(quant_cfg, compute_dtype):
    if not quant_cfg or not quant_cfg.get("enabled", False):
        return None
    bits = quant_cfg.get("bits", 4)
    if bits not in (4, 8):
        raise ValueError("Unsupported quantization bits: Use 4 or 8")
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
    )


# Pad-token candidates that some tokenizers already define in their vocab.
# These let us pick a pad token that is DIFFERENT from eos WITHOUT resizing the
# embedding matrix (resizing changes vocab size and adds untrained rows, which
# also pollutes the trainable-param counts that the magnitude audit reports).
# Order matters: most specific / most preferred first.
_PAD_TOKEN_CANDIDATES = (
    "<|finetune_right_pad_id|>",  # Llama 3.1 / 3.2 dedicated finetuning pad
    "<|pad|>",
    "<pad>",
    "[PAD]",
)


def _resolve_pad_token(tokenizer, model_cfg):
    """Choose a pad token that is distinct from eos whenever possible.

    The original DoRA / LLM-Adapters code deliberately set the pad id to a
    NON-eos token (`tokenizer.pad_token_id = 0  # unk`). We mirror that intent.

    Order of preference:
      1. An explicit pad token from config (model.pad_token).
      2. The tokenizer's own pad token, if it already has one distinct from eos.
      3. A reserved/existing token already in the vocab that isn't eos
         (no embedding resize needed).
      4. Fallback to pad_token = eos_token (with a warning) only if nothing
         else is available.

    Returns:
        bool: True if a brand-new token was added to the vocab (caller must
              then call model.resize_token_embeddings); False otherwise.
    """
    vocab = tokenizer.get_vocab()
    eos_id = tokenizer.eos_token_id

    # 1. Explicit override from config.
    configured = model_cfg.get("pad_token")
    if configured:
        if configured in vocab:
            tokenizer.pad_token = configured
            return False
        tokenizer.add_special_tokens({"pad_token": configured})
        print(f"[pad_token] added new pad token '{configured}' to the vocab (embeddings will be resized).")
        return True

    # 2. Tokenizer already has a usable pad token distinct from eos.
    if tokenizer.pad_token is not None and tokenizer.pad_token_id != eos_id:
        return False

    # 3. Reuse an existing in-vocab token distinct from eos (no resize).
    for candidate in _PAD_TOKEN_CANDIDATES:
        cand_id = vocab.get(candidate)
        if cand_id is not None and cand_id != eos_id:
            tokenizer.pad_token = candidate
            print(f"[pad_token] using existing vocab token '{candidate}' as pad (distinct from eos, no resize).")
            return False

    # 4. Last resort: fall back to eos and warn.
    tokenizer.pad_token = tokenizer.eos_token
    print(
        "[pad_token] WARNING: no distinct pad token found; falling back to "
        "pad_token = eos_token. Verify your data collator masks padding by "
        "POSITION (TRL's SFT collator does) so the real trailing EOS is still "
        "trained. If you use the transformers DataCollatorForLanguageModeling, "
        "the real EOS would be masked to -100 and the model would not learn to stop."
    )
    return False


def load_model_and_tokenizer(config):
    model_cfg = config["model"]
    training_cfg = config.get("training", {})
    quant_cfg = config.get("quantization", {})

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    torch_dtype = resolve_dtype(model_cfg.get("torch_dtype", "auto"))
    bnb_config = build_quantization_config(quant_cfg, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"],
        token=hf_token,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=model_cfg.get("use_fast_tokenizer", True),
    )
    # Resolve a pad token that stays distinct from eos when possible (matches the
    # original DoRA authors' deliberate choice of a non-eos pad). Only adds a new
    # token / requires a resize when an explicit, not-yet-present pad is configured.
    pad_token_added = _resolve_pad_token(tokenizer, model_cfg)
    tokenizer.padding_side = model_cfg.get("padding_side", "right")

    model_kwargs = {
        "token": hf_token,
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    if model_cfg.get("device_map"):
        model_kwargs["device_map"] = model_cfg["device_map"]
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_cfg["id"], **model_kwargs)

    # Only resize when a genuinely new token was added to the vocab.
    if pad_token_added:
        model.resize_token_embeddings(len(tokenizer))

    # Keep the model config's pad id aligned with the tokenizer so generation /
    # eval pad correctly. eos id is left untouched.
    model.config.pad_token_id = tokenizer.pad_token_id

    # use_cache must be off for gradient checkpointing compatibility.
    model.config.use_cache = False

    if bnb_config is not None:
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def load_training_datasets(dataset_cfg, tokenizer):
    fmt = dataset_cfg.get("format", "json")
    data_files = dataset_cfg["path"]
    split = dataset_cfg.get("split", "train")
    seed = dataset_cfg.get("seed", 42)
    validation_size = dataset_cfg.get("validation_size", 0)
    max_train_examples = dataset_cfg.get("max_train_examples")

    dataset = load_dataset(fmt, data_files=data_files, split=split)
    eos_token = tokenizer.eos_token or ""
    dataset = dataset.map(
        lambda example: format_example(example, eos_token),
        desc="Formatting commonsense examples",
    )

    if max_train_examples:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_train_examples, len(dataset))))

    if validation_size:
        split_dataset = dataset.train_test_split(test_size=validation_size, seed=seed)
        return split_dataset["train"], split_dataset["test"]
    return dataset, None


def build_lora_config(adapter_cfg):
    kwargs = {
        "r": adapter_cfg["rank"],
        "lora_alpha": adapter_cfg["alpha"],
        "target_modules": adapter_cfg["target_modules"],
        "lora_dropout": adapter_cfg.get("dropout", 0.0),
        "bias": adapter_cfg.get("bias", "none"),
        "task_type": adapter_cfg.get("task_type", "CAUSAL_LM"),
        "use_dora": adapter_cfg.get("use_dora", False),
    }
    if adapter_cfg.get("modules_to_save"):
        kwargs["modules_to_save"] = adapter_cfg["modules_to_save"]
    return LoraConfig(**kwargs)


def _filter_sft_kwargs(kwargs):
    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    if not fields:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in fields and v is not None}


def build_sft_config(config):
    experiment = config["experiment"]
    training = config["training"]
    kwargs = {
        "output_dir": experiment["output_dir"],
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "optim": training["optim"],
        "learning_rate": training["learning_rate"],
        "lr_scheduler_type": training["lr_scheduler_type"],
        "num_train_epochs": training["num_train_epochs"],
        "warmup_steps": training.get("warmup_steps", 0),
        "logging_steps": training["logging_steps"],
        "save_steps": training["save_steps"],
        "save_total_limit": training.get("save_total_limit"),
        "max_steps": training.get("max_steps", -1),
        "seed": training.get("seed", experiment.get("seed", 42)),
        "fp16": training.get("fp16", False),
        "bf16": training.get("bf16", False),
        "max_grad_norm": training["max_grad_norm"],
        "dataset_text_field": training.get("dataset_text_field", "text"),
        "max_length": training["max_length"],
        "packing": training.get("packing", False),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": training.get(
            "gradient_checkpointing_kwargs",
            {"use_reentrant": False},
        ),
        "report_to": training.get("report_to"),
        "logging_dir": training.get("logging_dir"),
        "eval_steps": training.get("eval_steps"),
        "eval_strategy": training.get("eval_strategy"),
        "save_strategy": training.get("save_strategy", "steps"),
        "dataloader_num_workers": training.get("dataloader_num_workers"),
    }
    return SFTConfig(**_filter_sft_kwargs(kwargs))


def build_callbacks(config, output_dir):
    monitoring = config.get("monitoring", {})
    metrics_dir = output_dir / "metrics"
    callbacks = []

    if monitoring.get("jsonl_logs", True):
        callbacks.append(JsonlLoggingCallback(metrics_dir / "trainer_log.jsonl"))

    resource_stats = monitoring.get("resource_stats", {})
    if resource_stats.get("enabled", True):
        callbacks.append(
            ResourceStatsCallback(
                metrics_dir / "resource_stats.jsonl",
                every_n_logs=resource_stats.get("every_n_logs", 1),
            )
        )

    analysis_snapshots = monitoring.get("analysis_snapshots", {})
    if analysis_snapshots.get("enabled", True):
        callbacks.append(
            AdapterSnapshotCallback(
                output_dir,
                fractions=analysis_snapshots.get("fractions", [0.25, 0.5, 0.75]),
                save_final=analysis_snapshots.get("save_final", True),
            )
        )

    gradient_stats = monitoring.get("gradient_stats", {})
    if gradient_stats.get("enabled", True):
        callbacks.append(
            GradientStatsCallback(
                metrics_dir / "gradient_stats.jsonl",
                every_n_steps=gradient_stats.get("every_n_steps", 50),
            )
        )
    return callbacks


def resolve_resume_checkpoint(output_dir, training_cfg):
    resume = training_cfg.get("resume_from_checkpoint", False)
    if not resume:
        return False
    if isinstance(resume, str) and resume not in {"true", "latest"}:
        return resume
    latest = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    if latest:
        return latest
    print("No checkpoints found; starting training from scratch.")
    return False


def write_run_metadata(config, output_dir, trainer, train_dataset, eval_dataset):
    model = trainer.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cuda_devices = []
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(device_idx)
            cuda_devices.append(
                {
                    "index": device_idx,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1024**3,
                    "major": props.major,
                    "minor": props.minor,
                }
            )

    metadata = {
        "experiment": config.get("experiment", {}),
        "model": config.get("model", {}),
        "training": config.get("training", {}),
        "adapter": config.get("adapter", {}),
        "cuda_devices": cuda_devices,
        "quantization": config.get("quantization", {}),
        "num_train_examples": len(train_dataset),
        "num_eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_param_percent": trainable_params / total_params * 100,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

def maybe_cast_magnitude_to_fp32(model, adapter_cfg):
    requested_dtype = adapter_cfg.get("magnitude_dtype")

    if requested_dtype not in {"fp32", "float32"}:
        print(
            f"[magnitude_dtype] native dtype retained; "
            f"requested={requested_dtype!r}"
        )
        return

    converted = []

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if _is_magnitude_param(name):
                parameter.data = parameter.data.to(torch.float32)
                converted.append(name)

    if not converted:
        raise RuntimeError(
            "Requested FP32 magnitude, but no DoRA magnitude "
            "parameters were found"
        )

    remaining = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if _is_magnitude_param(name)
        and parameter.dtype != torch.float32
    ]

    if remaining:
        raise RuntimeError(
            f"Magnitude parameters were not converted: {remaining[:5]}"
        )

    print(
        f"[magnitude_dtype] converted {len(converted)} "
        f"magnitude tensors to FP32"
    )

# ---------------------------------------------------------------------------
# Config finalization
# ---------------------------------------------------------------------------

def finalize_config(config):
    cfg = copy.deepcopy(config)
    experiment = cfg.setdefault("experiment", {})
    model = cfg.setdefault("model", {})
    adapter = cfg.setdefault("adapter", {})

    method = experiment.get("method") or infer_method(cfg)
    experiment["method"] = method

    rank = adapter.get("rank", "na")
    model_id = model.get("id", "model")
    model_short = sanitize_for_path(model_id.split("/")[-1])

    if not experiment.get("name"):
        template = experiment.get("run_name_template", "{model_short}-{method}-r{rank}")
        experiment["name"] = template.format(
            model_short=model_short,
            method=sanitize_for_path(str(method)),
            rank=rank,
        )

    if not experiment.get("output_dir"):
        output_root = experiment.get("output_root", "simple/outputs")
        experiment["output_dir"] = str(Path(output_root) / experiment["name"])

    return cfg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Single-file LoRA/DoRA/QLoRA/QDoRA trainer.")
    parser.add_argument("--config", default="./train_config.yaml")
    args = parser.parse_args()

    config = read_yaml(args.config)
    config = finalize_config(config)
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "resolved_config.yaml")

    model, tokenizer = load_model_and_tokenizer(config)
    train_dataset, eval_dataset = load_training_datasets(config["dataset"], tokenizer)
    peft_config = build_lora_config(config["adapter"])
    sft_args = build_sft_config(config)
    callbacks = build_callbacks(config, output_dir)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": sft_args,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "callbacks": callbacks,
    }
    try:
        trainer = SFTTrainer(**trainer_kwargs)
        
    except TypeError as exc:
        # Old TRL versions do not accept processing_class.
        if "processing_class" not in str(exc):
            raise
    
        trainer_kwargs.pop("processing_class")
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)
    
    
    # This must be OUTSIDE the try/except because both construction
    # paths require the intervention.
    maybe_cast_magnitude_to_fp32(
        trainer.model,
        config["adapter"],
    )
    
    
    # Fail fast when FP32 was explicitly requested.
    requested_dtype = config["adapter"].get("magnitude_dtype")
    
    if requested_dtype in {"fp32", "float32"}:
        magnitude_parameters = [
            (name, parameter)
            for name, parameter in trainer.model.named_parameters()
            if _is_magnitude_param(name)
        ]
    
        if not magnitude_parameters:
            raise RuntimeError(
                "FP32 magnitude requested, but no magnitude parameters were found"
            )
    
        actual_dtypes = {
            parameter.dtype
            for _, parameter in magnitude_parameters
        }
    
        if actual_dtypes != {torch.float32}:
            raise RuntimeError(
                f"FP32 magnitude intervention failed: {actual_dtypes}"
            )
    
        print(
            f"[magnitude_dtype] verified "
            f"{len(magnitude_parameters)} FP32 magnitude tensors"
        )
    write_run_metadata(config, output_dir, trainer, train_dataset, eval_dataset)

    run_magnitude_audit(
        trainer.model,
        output_dir / "metrics" / "magnitude_audit.json",
    )

    resume = resolve_resume_checkpoint(output_dir, config["training"])
    trainer.train(resume_from_checkpoint=resume)

    adapter_path = output_dir / "final_adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Saved final adapter to {adapter_path}")


if __name__ == "__main__":
    main()
