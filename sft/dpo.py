from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import json
from accelerate.utils import set_seed

set_seed(42)

from torch.utils.data import Dataset
from transformers import Trainer

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    beta: Optional[float] = field(default=0.1)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, data_path) -> Dataset:

    dataset = load_dataset('json', split=split, data_files=data_path)
    dataset = dataset.select(range(min(len(dataset), 1000)))
    
    def split_prompt_and_responses(sample) -> Dict[str, str]:
        print(sample)
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model_ref = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model_ref.eval()
    for param in model_ref.parameters():
        param.requires_grad = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = get_hh("train", data_args.data_path)

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()
    dpo_trainer.save_state()


if __name__ == "__main__":
    train()
