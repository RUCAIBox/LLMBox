from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class ImbuePrivate(MultipleChoiceDataset):
    """The dataset of Imbue private evaluations.

    High-quality question-answer pairs, from private versions of datasets designed to mimic ANLI, ARC, BoolQ, ETHICS, GSM8K, HellaSwag, OpenBookQA, MultiRC, RACE, Social IQa, and WinoGrande. For details, see https://imbue.com/research/70b-evals/. Format: each row contains a question, candidate answers, the correct answer (or multiple correct answers in the case of MultiRC questions), and a question quality score.

    Link: https://huggingface.co/datasets/imbue/high_quality_private_evaluations

    Example (To avoid data contamination, some fields are omitted):
        'question': 'For this question, first read the passage below. "The artist ..." Based on the passage above, answer the following question. Which wealth ...?'
        'correct_choices': [ "A ... ire" ]
        'choices': [ "A billionaire", "A centimillionaire", "A trillionaire", "A decamillionaire" ]
        'quality': 0.245109
        'original_dataset': race
    """

    instruction = "{{question}}{{'\n' + options if options}}\nAnswer:"
    evaluation_set = "train"
    example_set = None
    load_args = ("imbue/high_quality_private_evaluations",)
    category_column = "original_dataset"

    def format_instance(self, instance):
        if len(instance["correct_choices"]) > 1:
            logger.warning(
                f"Multiple correct choices found: {len(instance['correct_choices'])}. Only the first one is used. Multiple correct choices may be supported in the future."
            )

        correct_choice = instance["correct_choices"][0]
        instance["target_idx"] = instance["choices"].index(correct_choice)
        instance["options"] = instance["choices"]
        return instance

    @cached_property
    def references(self):
        return [instance["target_idx"] for instance in self.evaluation_data]
