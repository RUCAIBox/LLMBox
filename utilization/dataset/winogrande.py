from functools import cached_property
from logging import getLogger

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

    instruction = "{source}"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("winogrande", "winogrande_debiased")  # specify subset from command line

    def init_arguments(self):
        if self.ranking_with_options:
            logger.warning(
                f"Winogrande does not support ranking with options, automatically set ranking_with_options = False."
            )
            self.ranking_with_options = False

        if self.hf_prefix_caching:
            logger.warning(f"Winogrande does not support prefix_caching, automatically set prefix_caching = False.")
            self.hf_prefix_caching = False

    def format_instance(self, instance):
        question, completion = instance['sentence'].split('_')
        contexts = [question.strip() + ' ' + option for option in [instance['option1'], instance['option2']]]
        return dict(
            source=contexts,
            source_idx=int(instance["answer"]) - 1,
            target=completion,
        )

    @cached_property
    def references(self):
        return [int(instance["answer"]) - 1 for instance in self.evaluation_data]
