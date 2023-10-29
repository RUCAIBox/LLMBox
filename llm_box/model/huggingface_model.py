from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace

from .model import Model
from .utils import load_llm_and_tokenizer


class HuggingFaceModel(Model):

    def __init__(self, model_name_or_path: str, args: Namespace):
        super().__init__(args)
        self.model_name_or_path = model_name_or_path

        model, tokenizer = load_llm_and_tokenizer(args, model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer

    def get_ppl(self, batch):
        return super().get_ppl(batch)

    def generation(self, batch):
        return super().generation(batch)
