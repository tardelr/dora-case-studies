import argparse
import json
import os

from dotenv import load_dotenv
import torch

load_dotenv()
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# load config

def load_config(path: str = "train_config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# load dataset

def load_training_dataset(dataset_url: str):
    print("Loading dataset …")
    dataset = load_dataset("json", data_files=dataset_url, split="train")

    def format_example(example):
        parts = [example.get("instruction", "")]
        if example.get("input"):
            parts.append(example["input"])
        parts.append(example.get("output", ""))
        example["text"] = "\n".join(parts)
        return example

    dataset = dataset.map(format_example)
    print(f"Loaded {len(dataset)} examples")
    return dataset


# load model & tokenizer

def load_model(model_id: str, quant_cfg: dict = None, prepare_for_training: bool = True,
               use_bf16: bool = False):
    print("Loading model & tokenizer …")
    hf_token = os.environ.get("HF_TOKEN")
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = None
    if quant_cfg:
        bits = quant_cfg.get("bits", 4)
        if bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            )
        else:
            raise ValueError(f"Unsupported bits: {bits}. Use 4 or 8.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    if prepare_for_training and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


# lora/dora config

def get_peft_config(lora_cfg: dict, use_dora: bool = False):
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        use_dora=use_dora,
    )


# train

def train(model, tokenizer, dataset, peft_config, output_dir: str, train_cfg: dict):
    print("Starting training …")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        optim=train_cfg["optim"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_steps=train_cfg.get("warmup_steps", 0),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        max_steps=train_cfg.get("max_steps", -1),
        seed=train_cfg.get("seed", 42),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        max_grad_norm=train_cfg["max_grad_norm"],
        dataset_text_field="text",
        max_length=train_cfg["max_length"],
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )
    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint", False)
    if resume_from_checkpoint:
        import glob
        has_checkpoint = bool(glob.glob(os.path.join(output_dir, "checkpoint-*")))
        if not has_checkpoint:
            print("No checkpoints found — starting training from scratch.")
            resume_from_checkpoint = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


# save adapter

def save_adapter(trainer, tokenizer, output_dir: str, config: dict = None):
    adapter_path = f"{output_dir}/final_adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")
    if config is not None:
        config_path = f"{output_dir}/training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Training config saved to {config_path}")
    return adapter_path


# inference test

def test_inference(model, tokenizer, prompt: str):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


# apply adapter

def apply_adapter(base_model, adapter_path: str):
    print(f"Loading adapter from {adapter_path} …")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("Adapter loaded.")
    return model


# run benchmarks

def run_benchmarks(model, tokenizer, tasks, num_fewshot=0, limit=None, batch_size=16):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

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


# print results

def print_results(results: dict):
    print("\n── Results ──")
    for task, metrics in results["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc", "N/A"))
        acc_norm = metrics.get("acc_norm,none", metrics.get("acc_norm", "N/A"))
        print(f"  {task:>12s}  acc={acc}  acc_norm={acc_norm}")


# export results

def export_results(results: dict, config: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, default=str, indent=2)
    print(f"Results saved to {results_path}")

    config_path = os.path.join(output_dir, "eval_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


# training pipeline

def run_train(cfg):
    dataset = load_training_dataset(cfg["dataset_url"])
    use_bf16 = cfg.get("training", {}).get("bf16", False)
    model, tokenizer = load_model(cfg["model_id"], cfg["quantization"],
                                  use_bf16=use_bf16)
    peft_config = get_peft_config(cfg["lora"], use_dora=cfg["use_dora"])

    train_cfg = cfg["training"]
    if "seed" in cfg:
        train_cfg["seed"] = cfg["seed"]
    if "max_steps" in cfg:
        train_cfg["max_steps"] = cfg["max_steps"]
    if cfg.get("resume_from_checkpoint"):
        train_cfg["resume_from_checkpoint"] = cfg["resume_from_checkpoint"]

    trainer = train(model, tokenizer, dataset, peft_config, cfg["output_dir"], train_cfg)
    save_adapter(trainer, tokenizer, cfg["output_dir"], config=cfg)

    test_prompt = cfg.get("test_prompt")
    if test_prompt:
        test_inference(trainer.model, tokenizer, test_prompt)

    return trainer, model, tokenizer


# eval pipeline

def run_eval(cfg):
    eval_cfg = cfg["eval"]
    adapter_path = os.path.join(cfg["output_dir"], "final_adapter")

    use_bf16 = cfg.get("training", {}).get("bf16", False)
    model, tokenizer = load_model(cfg["model_id"], cfg.get("quantization"),
                                  prepare_for_training=False, use_bf16=use_bf16)

    if os.path.exists(adapter_path):
        model = apply_adapter(model, adapter_path)
    else:
        print(f"No adapter found at {adapter_path} — evaluating base model directly.")

    results = run_benchmarks(
        model,
        tokenizer,
        tasks=eval_cfg["tasks"],
        num_fewshot=eval_cfg.get("num_fewshot", 0),
        limit=eval_cfg.get("limit"),
        batch_size=eval_cfg.get("batch_size", 16),
    )
    print_results(results)

    results_dir = os.path.join(cfg["output_dir"], "eval-results")
    export_results(results, cfg, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate models with LoRA/DoRA")
    parser.add_argument("--config", default="train_config.json", help="Path to JSON config file")
    parser.add_argument("--mode", default="train", choices=["train", "eval", "all"],
                        help="Run mode: train, eval, or all (default: train)")
    parser.add_argument("--output", default=None, help="Output directory for adapter and eval results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output:
        cfg["output_dir"] = args.output

    if args.mode == "train":
        run_train(cfg)

    elif args.mode == "eval":
        run_eval(cfg)

    elif args.mode == "all":
        trainer, model, tokenizer = run_train(cfg)
        del trainer, model, tokenizer
        torch.cuda.empty_cache()
        run_eval(cfg)
