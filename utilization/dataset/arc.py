from logging import getLogger
from typing import List, Tuple, Union

import numpy as np

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

    instruction = ""
    evaluation_set = "test"
    example_set = "train"
    load_args = ("allenai/ai2_arc",)
    use_normalization = True
    normalization_prompt = "Question: \nAnswer:"

    def format_instance(self, instance):
        options = list(map(lambda _s: " " + _s, instance["choices"]["text"]))
        if instance["answerKey"].isdigit():
            instance["answerKey"] = ord(instance["answerKey"]) - 49
        else:
            instance["answerKey"] = ord(instance["answerKey"]) - 65
        return dict(
            source="Question: " + instance["question"],
            source_postfix="\nAnswer:",
            target_idx=instance["answerKey"],
            options=options,
        )

    def post_processing(self, predictions: List[Tuple[float, int]]) -> List[int]:
        labels = []
        st = 0
        predictions = list(map(lambda _r: _r[0], predictions))
        predictions = np.array([rc - ra for rc, ra in zip(predictions[::2], predictions[1::2])])
        for num in self.option_nums:
            labels.append(predictions[st:st + num].argmin())
            st += num
        predictions = labels
        return predictions

    @property
    def references(self):
        return [instance["answerKey"] for instance in self.evaluation_data]
