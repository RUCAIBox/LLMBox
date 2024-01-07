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
    
    name = 'wmt16'
    metric = "bleu"
    
    evaluation_set = "test"
    example_set = "train"
    metrics = [Bleu()]
    
    instruction = ''
    load_args = ('wmt16',)
    
    def __init__(self, args: DatasetArguments, model: Model, subset_name: Optional[str] = None):
        # pdb.set_trace()
        self.config = ' '
        self.config = subset_name
        raw_config = self.config if self.config[3:] == "en" else self.config[3:5] + "-" + self.config[:2]
        Wmt16.instruction = f"Translate from {self.config[:2]} to {self.config[3:5]}"
        # Wmt16.load_args = ("wmt16", raw_config)
        print(f"Init wmt16, config = {self.config}, raw_config = {raw_config}")
        
        super().__init__(args, model, subset_name)
    
    def format_instance(self, instance):
        # print(f'instance is {instance}')
        # print(f"in instance, config = {self.config}")
        instance = instance['translation']
        source_text = instance[self.config[:2]]
        target_text = instance[self.config[3:5]]
        return dict(source=source_text, target=target_text)

    @staticmethod
    def post_processing(predictions):
        return [pred.strip() for pred in predictions]
    
    @property
    def references(self):
        return [instance['translation'] for instance in self.evaluation_data]