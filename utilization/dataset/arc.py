from functools import cached_property
from logging import getLogger

from ..metric import Accuracy
from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class Arc(MultipleChoiceDataset):
    """The dataset of ai2_arc.

        A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage
        research in advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where
        the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence
        algorithm.

        Example:
            'id': 'Mercury_7175875',
            'question': 'An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?',
            'choices': {
                'text': ['Planetary density will decrease.', 'Planetary years will become longer.', 'Planetary days will become shorter.', 'Planetary gravity will become stronger.'],
                'label': ['A', 'B', 'C', 'D']
            },
            'answerKey': 'C'
        """

    instruction = "Question: {{question}}{{'\n' + options if options}}\nAnswer:"
    evaluation_set = "test"
    example_set = "train"
    load_args = ("allenai/ai2_arc",)
    use_normalization = True
    normalization_prompt = "Question: \nAnswer:"
    metrics = [Accuracy("Norm")]

    def init_arguments(self):
        # TODO
        self.prefix_caching = False

    def format_instance(self, instance):
        if instance["answerKey"].isdigit():
            instance["target_idx"] = ord(instance["answerKey"]) - ord("1")
        else:
            instance["target_idx"] = ord(instance["answerKey"]) - ord("A")
        return dict(
            question=instance["question"],
            target_idx=instance["target_idx"],
            options=instance["choices"]["text"],
        )

    @cached_property
    def references(self):
        return [instance["target_idx"] for instance in self.evaluation_data]
