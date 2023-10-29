from logging import getLogger
from typing import Optional, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..utils import ModelArguments

logger = getLogger(__name__)

OPENAI_MODELS = [
    'ada', 'babbage', 'curie', 'davinci', 'babbage-002', 'davinci-002', 'gpt-3.5-turbo', "gpt-3.5-turbo-instruct",
    "text-davinci-003"
]


def load_llm_and_tokenizer(
    model_name_or_path: str,
    args: ModelArguments,
    tokenizer_name_or_path: Optional[str] = None,
) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:

    model_kwargs = dict(
        torch_dtype=torch.float16 if args.load_in_half else torch.float32,
        device_map=args.device_map,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        padding_side='left',
    )

    # set `pad` token to `eos` token
    model.config.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
