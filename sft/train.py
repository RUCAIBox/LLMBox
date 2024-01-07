import warnings
import logging
import torch

from typing import Optional
from dataclasses import dataclass, MISSING
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser
from transformers.hf_argparser import HfArg
from datasets import load_dataset
from accelerate.utils import set_seed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from dataset import Dataset


@dataclass
class Arguments(TrainingArguments):

    model_name_or_path: str = HfArg(
        default=MISSING, help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf` or `./output/saved_model`"
    )

    tokenizer_name_or_path: Optional[str] = HfArg(
        default=None,
        help="The tokenizer name or path. Default to `model_name_or_path`.",
    )

    data_path: str = HfArg(
        default=MISSING, help="The path of dataset, e.g., `data/alpaca_data.json.` or `data/chinese.txt`"
    )

    model_max_length: int = HfArg(
        default=2048,
        help="The maximum sequence length",
    )

    mode: str = HfArg(
        default='sft',
        help="The mode of the training programs, which must be chosen from either `sft` or `pt`.",
    )

    use_flash_attention: bool = HfArg(
        default=True,
        help="When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
    )

    save_only_model: bool = HfArg(
        default=True,
        help=
        "Whether to use flash attention for a faster and more efficient implementation of the standard attention mechanism."
    )

    bf16: bool = HfArg(
        default=True,
        help="Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
        " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
    )

    tf32: Optional[bool] = HfArg(
        default=True,
        help="Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
        " API and it may change."
    )


def train():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=True,
        use_fast=False,
        legacy=False,  # refer to the issue:https://github.com/huggingface/transformers/pull/24565
        use_cache=False,
    )
    tokenizer.pad_token = tokenizer.unk_token  # for llama-1

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        use_cache=False,  # When gradient checkpointing used, set this to False
        attn_implementation="flash_attention_2" if args.use_flash_attention else None
    )

    if args.mode == 'sft':
        dataset = Dataset(args)

        # set the template for the instruction and response
        instruction_template_ids = tokenizer.encode(dataset.instruction_template, add_special_tokens=False)[1:]
        response_template_ids = tokenizer.encode(dataset.response_template, add_special_tokens=False)[1:]

        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template_ids,
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset.load_data(),
            tokenizer=tokenizer,
            max_seq_length=args.model_max_length,
            packing=False,
            data_collator=collator,
            formatting_func=dataset.formatting_func,
        )
    elif args.mode == 'pt':
        dataset = load_dataset('text', data_files=args.data_path)['train']
        model.resize_token_embeddings(len(tokenizer))

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=args.model_max_length,
            packing=True,
            dataset_text_field="text"
        )

    trainer.train()
    trainer.save_state()


def init():
    set_seed(42)
    warnings.filterwarnings("ignore")
    logging.getLogger("DeepSpeed").setLevel(logging.ERROR)


if __name__ == "__main__":
    init()
    train()
