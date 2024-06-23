import os
import sys

import torch
from transformers import LlamaForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utilization import DatasetArguments, ModelArguments, get_evaluator


def load_hf_model(model_args: ModelArguments):
    from utilization.model.huggingface_model import get_model_max_length, load_tokenizer

    # load your own model
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        device_map=model_args.device_map,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        trust_remote_code=True,
    )

    max_length = get_model_max_length(model)
    tokenizer = load_tokenizer(model_args.tokenizer_name_or_path, use_fast=True, max_length=max_length)

    return model, tokenizer


evaluator = get_evaluator(
    model_args=ModelArguments(
        model_name_or_path="../your-model-path",
        model_type="chat",
        model_backend="huggingface",
        prefix_caching=False,
        vllm=False,
        max_tokens=300,
        torch_dtype="auto",
    ),
    dataset_args=DatasetArguments(
        dataset_names=["mmlu"],
        batch_size=1,
        num_shots=5,
        max_example_tokens=2560,
    ),
    load_hf_model=load_hf_model,
)
evaluator.evaluate()
