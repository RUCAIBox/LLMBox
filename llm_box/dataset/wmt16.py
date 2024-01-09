from typing import Optional
import numpy as np

from llm_box.model.model import Model
from llm_box.utils import DatasetArguments

from .generation_dataset import GenerationDataset
from ..metric import Bleu

class Wmt16(GenerationDataset):
    """ The dataset of Wmt dataset.
    
    Example:
        config: ro-en
        instance: {'translation': {'en': 'Obama welcomes Netanyahu', 'ro': 'Obama primește Netanyahu'}
        prediction: Obama receives Netanyahu 
        reference: Obama welcomes Netanyahu
    """
    
    name = 'wmt16'
    metric = "bleu"
    
    evaluation_set = "test"
    example_set = "train"
    metrics = [Bleu()]
    
    instruction = ''
    load_args = ('wmt16',)
    
    def __init__(self, args: DatasetArguments, model: Model, subset_name: Optional[str] = None):
        self.config = subset_name
        # raw_config = self.config if self.config[3:] == "en" else self.config[3:5] + "-" + self.config[:2]
        # print(f"Init wmt16, config = {self.config}, raw_config = {raw_config}")
                
        super().__init__(args, model, subset_name)
    
    def format_instance(self, instance):
        language = {
            "en": "English",
            "de": "German",
            "fr": "French",
            "cs": "Czech",
            "fi": "Finnish",
            "ru": "Russian",
            "tr": "Turkish",
            "ro": "Romanian"
        }
        instance = instance['translation']
        source_text = f"Q: What is the {language[self.config[3:5]]} translation of {instance[self.config[:2]]} A:"
        target_text = instance[self.config[3:5]]
        return dict(source=source_text, target=target_text)

    @staticmethod
    def post_processing(predictions):
        return [pred.strip() for pred in predictions]
    
    @property
    def references(self):
        return [instance['translation'][self.config[3:]] for instance in self.evaluation_data]