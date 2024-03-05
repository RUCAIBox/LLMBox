from ..metric import Alpaca_judge
from .generation_dataset import GenerationDataset
import re


class Alpaca_farm(GenerationDataset):
    """The dataset of Alpaca_farm dataset.

    Example:
        instruction: How did US states get their names?
        output: Most US states were named after either Native American tribes, geographical features, or historical figures. For example, the state of Florida was named after the Spanish explorer Ponce de Leon, and the state of Texas was named after the Caddo word “tejas” meaning friends or allies.
    """

    instruction = "Please answer the following question."
    evaluation_set = "eval"
    example_set = "eval"
    metrics = [Alpaca_judge()]
    load_args = ("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")
    extra_model_args = dict(temperature=0, max_tokens=128)

    def format_instance(self, instance):
        return dict(source=instance["instruction"], target="")

    @property
    def references(self):
        return self.evaluation_data
