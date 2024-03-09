import os
import logging
import warnings
from dataclasses import dataclass
from typing import Optional
from accelerate.utils import set_seed
from sft_dataset import AutoDataset
from pt_dataset.pt_dataset import PTDataset
from peft import LoraConfig, TaskType, AutoPeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from transformers.hf_argparser import HfArg
from typing import Dict, Optional, Sequence
import torch
import transformers
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)
IGNORE_INDEX = -100
@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf` or `./output/saved_model`",
    )

    tokenizer_name_or_path: Optional[str] = HfArg(
        default=None,
        help="The tokenizer name or path. Default to `model_name_or_path`.",
    )

    data_path: str = HfArg(
        default=None,
        help="The path of dataset, e.g., `data/alpaca_data.json.` or `data/chinese.txt`",
    )

    model_max_length: int = HfArg(
        default=2048, 
        help="The maximum sequence length",
    )

    mode: str = HfArg(
        default="sft",
        help="The mode of the training programs, which must be chosen from either `sft` or `pt`.",
        metadata={"choices": ["sft", "pt"]},
    )

    save_only_model: bool = HfArg(
        default=True,
        help="When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.",
    )

    use_flash_attention: bool = HfArg(
        default=True,
        help=
        "Whether to use flash attention for a faster and more efficient implementation of the standard attention mechanism.",
    )

    rope_scaling_type: str = HfArg(
        default = "none",
        help="Whether to scaling the RoPE. `none` denotes no scaling of RoPE . `dynamic` and `linear` denoted to scaling RoPE with dynamic NTK and Position Interpolation."
    )

    rope_scaling_factor: int = HfArg(
        default = 4,
        help="Scaling factor of RoPE. The maximum context length will be expanded to the factor times the original maximum positional embedding length."
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

    lora: Optional[bool] = HfArg(default=False, help="whether to train with LoRA.")

    lora_r: Optional[int] = HfArg(default=16, help='Lora attention dimension (the "rank")')

    lora_alpha: Optional[int] = HfArg(default=16, help="The alpha parameter for Lora scaling.")

    lora_dropout: Optional[float] = HfArg(default=0.05, help="The dropout probability for Lora layers.")
    
    lora_target_modules: Optional[str] = HfArg(default=None, help="The target modules for LoRA training. e.g. `q_proj,k_proj`.")
    
    lora_merge: Optional[bool] = HfArg(default=True, help="Whether to merge the LoRA model after training.")
    
    qlora: Optional[bool] = HfArg(default=False, help="whether to train with QLoRA. This will enable LoRA automatically.")
    
    packing: Optional[bool] = HfArg(default=False, help="Whether to pack the input sequences to the maximum length of the model.")


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
        )

def train():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    
    if args.gradient_checkpointing:
        args.gradient_checkpointing_kwargs={'use_reentrant':False} # OR gradient_checkpointing_kwargs={'use_reentrant':True}, please refer to https://github.com/huggingface/transformers/issues/26969

    if args.qlora:
        args.lora = True

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if args.rope_scaling_type != "none":
        config.rope_scaling = {
            "type" : args.rope_scaling_type,
            "factor" : args.rope_scaling_factor
        }
        
    if args.use_flash_attention:
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation = None
    config.use_cache=False
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=True,
        # use_fast=False, # some tokenizer has only one implementation
        legacy=False,  # refer to the issue:https://github.com/huggingface/transformers/pull/24565
        use_cache=False,
    )
    
    tokenizer.pad_token = tokenizer.unk_token  # for llama-1
    
    if config._attn_implementation == "flash_attention_2" or args.qlora or args.bf16:
        load_type =  torch.bfloat16 
    else:
        load_type = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=load_type,
        config=config,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ) if args.qlora else None
    ) 
    if args.lora:
        if args.qlora:
            model = prepare_model_for_kbit_training(model, args.lora_r)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
        )
        model = get_peft_model(model, peft_config)
    
    kwargs = dict(
        model=model,
        args=args,
        tokenizer=tokenizer,
    )
    if args.mode == "sft":
        kwargs.update(
            dict(
                train_dataset=AutoDataset(args, tokenizer),
                data_collator=DataCollatorForSupervisedDataset(tokenizer),
            )
        )

    elif args.mode == "pt":
        model.resize_token_embeddings(len(tokenizer))
        kwargs.update(
            dict(
                train_dataset=PTDataset(args, tokenizer),
                data_collator=DataCollatorForSupervisedDataset(tokenizer),
            )
        )

    trainer = Trainer(**kwargs)
    trainer.train()
    trainer.save_model(args.output_dir+"/checkpoint-final")
    trainer.save_state()
    
    if args.lora and args.lora_merge:
        if is_deepspeed_zero3_enabled():
            unset_hf_deepspeed_config()
        subdir_list = os.listdir(args.output_dir)
        for subdir in subdir_list:
            if subdir.startswith("checkpoint"):
                print("Merging model in ", args.output_dir+"/"+subdir)
                peft_model = AutoPeftModelForCausalLM.from_pretrained(args.output_dir+"/"+subdir)
                merged_model = peft_model.merge_and_unload()
                save_path = args.output_dir+"/"+subdir+"-merged"
                merged_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)


def init():
    set_seed(42)
    warnings.filterwarnings("ignore")
    logging.getLogger("DeepSpeed").setLevel(logging.ERROR)


if __name__ == "__main__":
    init()
    train()
