import re
from functools import cached_property

import numpy as np

from ..metric import Accuracy
from .generation_dataset import GenerationDataset


class Color_objects(GenerationDataset):
    """The dataset of BIG-Bench Reasoning_about_color_objects dataset.

    BIG-Bench but it doesn't require the hellish dependencies (tensorflow, pypi-bigbench, protobuf) of the official version.

    Example:
        inputs: Q: On the floor, I see three silver keychains, three burgundy keychains, three burgundy teddy bears, three magenta teddy bears, three burgundy stress balls, and three magenta keychains. If I remove all the burgundy items from the floor, how many keychains remain on it? A:
        targets: [ "6" ]
        multiple_choice_targets: [ "0", "zero", "1", "one", "2", "two", "3", "three", "4", "four", "5", "five", "6", "six", "7", "seven", "8", "eight", "9", "nine", "10", "ten", "11", "eleven", "12", "twelve", "13", "thirteen", "14", "fourteen", "15", "fifteen", "16", "sixteen" ]
        multiple_choice_scores: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    """

    evaluation_set = "validation"
    example_set = "train"
    metrics = [Accuracy()]
    instruction = "{inputs}"
    load_args = ("tasksource/bigbench", "reasoning_about_colored_objects")
    extra_model_args = dict(temperature=0, stop=["\n"])

    def init_arguments(self):
        # TODO fix color_objects prefix_caching
        self.hf_prefix_caching = False

    def format_instance(self, instance):
        return dict(inputs=instance["inputs"], target=instance["targets"][0])

    def post_processing(self, predictions):
        new_predictions = []
        pattern = r"[.,!(\n)]"
        for pred, instance in zip(predictions, self.evaluation_data):
            pred = pred.lower().strip().split("\n")[0]
            match = re.search(pattern, pred)
            if match:
                index = match.start()
                pred = pred[:index]
            refer = np.array(instance["multiple_choice_targets"])
            idx = np.array(instance["multiple_choice_scores"])
            refer = refer[np.where(idx == 1)[0]]
            new_predictions.append(True if pred in refer else False)
        return new_predictions

    @cached_property
    def references(self):
        return [True for _ in self.evaluation_data]
