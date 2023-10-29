from logging import getLogger
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = getLogger(__name__)

OPENAI_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'babbage-002', 'davinci-002', 'gpt-3.5-turbo']


def load_llm_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
):

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path)

    # set `pad` token to `eos` token
    if hasattr(model.config, 'eos_token'):
        model.config.pad_token = model.config.eos_token
    if hasattr(model.config, 'eos_token_id'):
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
