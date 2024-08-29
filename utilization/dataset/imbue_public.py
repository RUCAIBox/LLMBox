from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class ImbuePublic(MultipleChoiceDataset):
    """The dataset of Imbue public evaluations.

    High-quality question-answer pairs, originally from ANLI, ARC, BoolQ, ETHICS, GSM8K, HellaSwag, OpenBookQA, MultiRC, RACE, Social IQa, and WinoGrande. For details, see https://imbue.com/research/70b-evals/. Format: each row contains a question, candidate answers, the correct answer (or multiple correct answers in the case of MultiRC questions), and a question quality score.

    Link: https://huggingface.co/datasets/imbue/high_quality_public_evaluations

    Example:
        'question': 'The man was released from jail. What is the cause of this?'
        'correct_choices': [ "His family paid his bail." ]
        'choices': [ "His family paid his bail.", "He attacked a fellow inmate." ]
        'quality': 0.348698
        'original_dataset': copa
    """

    instruction = "{{question}}{{'\n' + options if options}}\nAnswer:"
    evaluation_set = "train"
    example_set = None
    load_args = ("imbue/high_quality_public_evaluations", )
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
