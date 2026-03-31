import argparse
import json
import os

from dotenv import load_dotenv
import torch

load_dotenv()
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ── Load config 

def load_config(path: str = "train_config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── Load dataset 
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


# ── Load model & tokenizer ──────────────────────────────────────────

def load_model(model_id: str, quant_cfg: dict):
    print("Loading model & tokenizer …")
    hf_token = os.environ.get("HF_TOKEN")
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
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


# ── Configure LoRA/DoRA adapter ──────────────────────────────────────

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


# ── Train 

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
        bf16=train_cfg.get("bf16", False),
        max_grad_norm=train_cfg["max_grad_norm"],
        dataset_text_field="text",
        max_length=train_cfg["max_length"],
        packing=True,
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


# ── Save adapter 
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


# ── inference test 

def test_inference(model, tokenizer, prompt: str):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


# ── Main 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA/DoRA on commonsense data")
    parser.add_argument("--config", default="train_config.json", help="Path to JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset = load_training_dataset(cfg["dataset_url"])
    model, tokenizer = load_model(cfg["model_id"], cfg["quantization"])
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
