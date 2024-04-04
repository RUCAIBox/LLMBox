import re

from ..metric import F1, Em
from .generation_dataset import GenerationDataset


class WebQ(GenerationDataset):
    """The dataset of Web Questions.

    This dataset consists of 6,642 question/answer pairs.
    The questions are supposed to be answerable by Freebase, a large knowledge graph.
    The questions are mostly centered around a single named entity.
    The questions are popular ones asked on the web (at least in 2013).

    Examples:
        'answers': ['Jamaican Creole English Language', 'Jamaican English'],
        'question': 'what does jamaican people speak?',
        'url': 'http://www.freebase.com/view/en/jamaica'
    """

    instruction = ""
    example_set = "train"
    evaluation_set = "test"
    load_args = ("web_questions",)
    metrics = [F1(), Em()]
    extra_model_args = dict(max_tokens=64, temperature=0, stop=["\n"])

    def format_instance(self, instance):
        source_text = "Q: " + instance["question"] + "\n\nA:"
        target_text = " " + instance["answers"][0]
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
        return [instance["answers"] for instance in self.evaluation_data]
