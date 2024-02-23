from logging import getLogger

import numpy as np

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class WinoGrande(MultipleChoiceDataset):
    """The dataset of WinoGrande.

    WinoGrande is a new collection of 44k problems, inspired by Winograd Schema Challenge
    (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and robustness against the
    dataset-specific bias. Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right
    option for a given sentence which requires commonsense reasoning.

    Example:
        'answer': '2',
        'option1': 'Sarah',
        'option2': 'Maria',
        'sentence': 'Sarah was a much better surgeon than Maria so _ always got the easier cases.'
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("winogrande", "winogrande_debiased")  # specify subset from command line

    def format_instance(self, instance):
        question, postfix = instance['sentence'].split('_')
        options = [instance["option1"] + postfix, instance["option2"] + postfix]
        return dict(
            source=question.strip(),
            target_idx=int(instance["answer"]) - 1,
            options=options,
        )

    @property
    def references(self):
        return [int(instance["answer"]) - 1 for instance in self.evaluation_data]
