import re

from ..metric import F1, Em
from .generation_dataset import GenerationDataset


class Nq(GenerationDataset):
    """The dataset of Nq.

    The NQ-Open task, introduced by Lee et.al. 2019, is an open domain question answering benchmark that is derived from Natural Questions.
    The goal is to predict an English answer string for an input English question.
    All questions can be answered using the contents of English Wikipedia.

    Examples:
        'answer': ['14 December 1972 UTC', 'December 1972'],
        'question': 'when was the last time anyone was on the moon'
    """

    example_set = "train"
    evaluation_set = "validation"
    load_args = ("nq_open",)
    metrics = [F1(), Em()]
    extra_model_args = dict(max_tokens=64, temperature=0, stop=["\n"])

    def _format_instance(self, instance):
        source_text = "Q: " + instance["question"] + "\n\nA:"
        target_text = " " + instance["answer"][0]
        return dict(source=source_text, target=target_text)

    @staticmethod
    def post_processing(predictions):
        new_predictions = []
        pattern = r"[.!(\n)]"
        for pred in predictions:
            match = re.search(pattern, pred)
            if match:
                index = match.start()
                pred = pred[:index]
            new_predictions.append(pred)
        return new_predictions

    @property
    def references(self):
        return [instance["answer"] for instance in self.evaluation_data]
