
# for each benchmark (boolq, hellaswag, run this process)
import argparse
import json
import os

# from dotenv import load_dotenv
# import torch

# load_dotenv()
# from datasets import load_dataset
from transformers import AutoTokenizer
# from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
# from trl import SFTTrainer, SFTConfig

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.eos_token)
# import lm_eval
# from lm_eval.models.huggingface import HFLM

eval_cfg = "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/ft-training_set/commonsense_170k.json"
from datasets import load_dataset


def load_training_dataset(dataset_url: str):
    print("Loading dataset …")
    dataset = load_dataset("json", data_files=dataset_url, split="train[:20]")

    def format_example(example):
        if example["input"]:
            example['text'] = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {example["instruction"]}
                    
                    ### Input:
                    {example["input"]}
                    
                    ### Response:
                    {example["output"]}"""
        else:
            example['text'] = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                    ### Instruction:
                    {example["instruction"]}
                    
                    ### Response:
                    {example["output"]}""" 
        example['text'] += tokenizer.eos_token
        print(example)
        return example

    dataset = dataset.map(format_example)
    
    print(f"Loaded {len(dataset)} examples")
    return dataset

dataset = load_training_dataset(eval_cfg)


