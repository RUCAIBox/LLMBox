from functools import cached_property
from logging import getLogger

from ..dataset_enum import MMLU_SUBJECTS
from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class Mmlu(MultipleChoiceDataset):
    """The dataset of MMLU.

    Measuring Massive Multitask Language Understanding by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).

    Example:
        "question": "What is the embryological origin of the hyoid bone?",
        "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
        "answer": 3
    """

    instruction = "The following are multiple choice questions (with answers) about {{subset}}.\n\nQuestion: {{source}}{{ '\n' + options if options }}\nAnswer:"
    evaluation_set = "test"
    example_set = "dev"
    load_args = ("hails/mmlu_no_train",)
    categorized_subsets = MMLU_SUBJECTS
    banned_subsets = ["all"]

    @staticmethod
    def _format_subject(subject: str) -> str:
        return subject.replace("_", " ")

    def format_instance(self, instance):
        return dict(
            source=instance["question"].strip(),
            target_idx=instance["answer"],
            subset=self._format_subject(self.subset_name),
            options=instance["choices"],
        )

    @cached_property
    def references(self):
        return [instance["answer"].strip() for instance in self.evaluation_data]
