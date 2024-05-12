from functools import cached_property

from ..metric import GPTEval
from .generation_dataset import GenerationDataset


class Alpaca_eval(GenerationDataset):
    """The dataset of Alpaca_farm dataset.

    Example:
        instruction: How did US states get their names?
        output: Most US states were named after either Native American tribes, geographical features, or historical figures. For example, the state of Florida was named after the Spanish explorer Ponce de Leon, and the state of Texas was named after the Caddo word “tejas” meaning friends or allies.
    """

    instruction = "Please answer the following question."
    evaluation_set = "eval"
    example_set = None
    metrics = [GPTEval(multi_turn=False, type="pairwise")]
    load_args = ("tatsu-lab/alpaca_eval", "alpaca_eval")
    extra_model_args = dict(temperature=0.7, max_tokens=1024)

    def format_instance(self, instance):
        return dict(source=instance["instruction"], target="")

    @staticmethod
    def post_processing(predictions):
        return [prediction.strip() for prediction in predictions]

    @cached_property
    def references(self):
        return self.evaluation_data
