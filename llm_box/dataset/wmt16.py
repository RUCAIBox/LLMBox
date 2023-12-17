from typing import Optional
import numpy as np

from llm_box.model.model import Model
from llm_box.utils import DatasetArguments

from .generation_dataset import GenerationDataset
from ..metric import Bleu

class Wmt16(GenerationDataset):
    """ The dataset of Wmt dataset.
    
    Example:
        hypothesis: Obama welcomes Netanyahu
        reference: Obama receives Netanyahu
    """
    def __init__(self, config, args: DatasetArguments, model: Model, subset_name: str | None = None):
        super().__init__(args, model, subset_name)
        self.config = config
        instruction = f"Translate from {self.config[:2]} to {self.config[3:5]}"
        load_args = ("wmt16", self.config)
    
    name = 'wmt16'
    metric = "bleu"
    
    evaluation_set = "test"
    example_set = "train"
    metrics = [Bleu()]
    
    def format_instance(self, instance):
        source_text = instance[self.config[:2]]
        target_text = instance[self.config[3:5]]
        return dict(source=source_text, target=target_text)

    @staticmethod
    def post_processing(predictions):
        return [pred.strip() for pred in predictions]
    
    @property
    def references(self):
        return [instance['translation'] for instance in self.evaluation_data]