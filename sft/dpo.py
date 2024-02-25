from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import logging
import warnings
import torch
import transformers
from accelerate.utils import set_seed

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.hf_argparser import HfArg

from trl import DPOTrainer


@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf` or `./output/saved_model`",
    )

    data_path: str = HfArg(
        default=None,
        help="The path of preference dataset, e.g., `Anthropic/hh-rlhf`, `Dahoas/rm-static` or `Dahoas/synthetic-instruct-gptj-pairwise`",
    )
    
    model_max_length: int = HfArg(
        default=512,
        help="Maximum sequence length. Sequences will be right padded (and possibly truncated)."
    )

    bf16: bool = HfArg(
        default=True,
        help="Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
        " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change.",
    )

    tf32: Optional[bool] = HfArg(
        default=True,
        help="Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
        " API and it may change.",
    )
    
    cache_dir: str = HfArg(default=None)
    
    beta: float = HfArg(
        default=0.1,
        help="The beta factor in DPO loss."
        "Higher beta means less divergence from the initial policy.",
    )
    
    loss_type: str = HfArg(
        default="sigmoid",
        help="The type of DPO loss to use, e.g., `sigmoid`, `hinge` or `ipo`",
    )



def get_data(split: str, data_path) -> Dataset:

    dataset = load_dataset(split=split, path=data_path)
    
    def split_prompt_and_responses_hh(sample) -> Dict[str, str]:
        search_term = "\n\nAssistant:"
        search_term_idx = sample["chosen"].rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        prompt = sample["chosen"][: search_term_idx + len(search_term)]
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }
    
    def split_prompt_and_responses_rm(sample) -> Dict[str, str]:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }

    def split_prompt_and_responses_syn(sample) -> Dict[str, str]:
        return {
            "prompt": "Human: " + sample["prompt"] + "Assistant: ",
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
    
    if 'hh-rlhf' in data_path:
        return dataset.map(split_prompt_and_responses_hh)
    if 'rm-static' in data_path:
        return dataset.map(split_prompt_and_responses_rm)
    if 'synthetic' in data_path:
        return dataset.map(split_prompt_and_responses_syn)

def train():

    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model_ref = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model_ref.eval()
    for param in model_ref.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=True,
        use_fast=False,
        legacy=False,  # refer to the issue:https://github.com/huggingface/transformers/pull/24565
        use_cache=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = get_data("train", args.data_path)
    
    kwargs = dict(
        model=model,
        ref_model=model_ref,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    dpo_trainer = DPOTrainer(**kwargs)

    dpo_trainer.train()
    dpo_trainer.save_state()


def init():
    set_seed(42)
    warnings.filterwarnings("ignore")
    logging.getLogger("DeepSpeed").setLevel(logging.ERROR)


if __name__ == "__main__":
    init()
    train()
