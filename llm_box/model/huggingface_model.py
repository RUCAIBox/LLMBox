from argparse import Namespace

from .model import Model
from .utils import load_raw_model


class HuggingFaceModel(Model):

    name = None

    def __init__(self, model_name_or_path: str, args: Namespace):
        super().__init__(args)
        self.model_name_or_path = model_name_or_path

        model = load_raw_model(args, model_name_or_path)
        self.model = model
        self.name = model._get_name()
        self.tokenizer = args.tokenizer

    def get_ppl(self, batched_inputs):
        pass

    def generation(self, batched_inputs):
        return super().generation(batched_inputs)
