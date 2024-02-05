import numpy as np

from .multiple_choice_dataset import MultipleChoiceDataset


class Mmlu(MultipleChoiceDataset):
    """The dataset of MMLU.

    Measuring Massive Multitask Language Understanding by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).

    Example:
        "question": "What is the embryological origin of the hyoid bone?",
        "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
        "answer": 3
    """

    instruction = "The following are multiple choice questions (with answers) about {}."
    evaluation_set = "test"
    example_set = "dev"
    load_args = ("hails/mmlu_no_train", )  # remove "all" by default

    def __init__(self, args, model, subset_name=None):
        self.instruction = self.instruction.format(subset_name)
        super().__init__(args, model, subset_name)

    def format_instance(self, instance):
        options = list(map(lambda op: " " + op, instance["choices"]))
        return dict(
            source=instance["question"],
            target=options[instance["answer"]],
            options=options,
        )

    def calculate_metric(self, predictions):
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @property
    def references(self):
        return [instance["answer"] for instance in self.evaluation_data]
