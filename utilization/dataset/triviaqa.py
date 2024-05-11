import re
from functools import cached_property

from ..metric import F1, Em
from .generation_dataset import GenerationDataset


class TriviaQA(GenerationDataset):
    """The dataset of TriviaQA.

    TriviaqQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions.

    Examples:
        'answer': {'aliases': ['Sunset Blvd',
                        'West Sunset Boulevard',
                        'Sunset Boulevard',
                        'Sunset Bulevard',
                        'Sunset Blvd.'],
            'normalized_aliases': ['sunset boulevard',
                                   'sunset bulevard',
                                   'west sunset boulevard',
                                   'sunset blvd'],
            'normalized_value': 'sunset boulevard',
            'value': 'Sunset Boulevard'},
        'question': 'Which Lloyd Webber musical premiered in the US on 10th December 1993?',
    """

    instruction = "Q: {question}\n\nA:"
    example_set = "train"
    evaluation_set = "validation"
    load_args = ("trivia_qa", "rc.wikipedia.nocontext")
    metrics = [F1(), Em()]
    extra_model_args = dict(max_tokens=64, temperature=0, stop=["\n"])

    def format_instance(self, instance):
        instance["target"] = instance["answer"]["value"]
        return instance

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

    @cached_property
    def references(self):
        return [
            instance["answer"]["normalized_aliases"] + [instance["answer"]["normalized_value"]]
            for instance in self.evaluation_data
        ]
