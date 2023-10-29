from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
from argparse import Namespace

from ..utils import args_to_tokenizer_kwargs, args_to_model_kwargs

OPENAI_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'babbage-002', 'davinci-002']


def load_tokenizer(
    tokenizer_name_or_path: str,
    copy_special_tokens: Dict[str, str] = None,
    tokenizer_kwargs: Dict[str, Any] = None,
):
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        **tokenizer_kwargs
    )

    if copy_special_tokens is not None:
        for tgt, src in copy_special_tokens.items():
            tgt_token = tgt + "_token"
            tgt_token_id = tgt + "_token_id"
            src_token = src + "_token"
            src_token_id = src + "_token_id"
            if getattr(tokenizer, tgt_token, None) is None:
                setattr(tokenizer, tgt_token, src_token)
                setattr(tokenizer, tgt_token_id, src_token_id)

    return tokenizer


def load_llm_and_tokenizer(
    args: Namespace,
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
):

    if getattr(args, "gptq", False):
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model

    else:
        model_kwargs = args_to_model_kwargs(args)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
    model.eval()

    tokenizer_kwargs = args_to_tokenizer_kwargs(args)
    tokenizer = load_tokenizer(
        tokenizer_name_or_path or model_name_or_path,
        # set `pad` token to `eos` token if pad token is not set
        # (as is the case for llama models)
        {'pad': 'eos'},
        tokenizer_kwargs,
    )

    return model, tokenizer

