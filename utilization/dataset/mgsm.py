import re
import signal
from functools import cached_property

from ..metric import Accuracy
from .generation_dataset import GenerationDataset


class Mgsm(GenerationDataset):
    """The dataset of MGSM.

    Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems.

    Examples:
        'question': 'Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?',
        'answer': 'Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.',
        'answer_number': 11,
        'equation_solution': '5 + 6 = 11.'
    """
    
    instruction = "Answer the following question in {{lang}}.\n\nQuestion: {{question.replace('\n', ' ')}}\nAnswer:"
    
    evaluation_set = "test"
    example_set = "train"
    load_args = ("juletxara/mgsm",)
    metrics = [Accuracy()]
    extra_model_args = dict(temperature=0)

    _decimal_separator = re.compile(r"(\d),(\d)")
    _extract_numbers = re.compile(r"[-+]?\d*\.\d+|\d+")

    def init_arguments(self):
        if self.model_type == 'base':
            self.extra_model_args['stop'] = ['\n']

        from langcodes import Language
        self.language = Language(self.subset_name).language_name("en")

    def post_processing(self, predictions):
        new_predictions = []
        for pred in predictions:
            # replace numbers like `x,xxx` with `xxxx`
            pred = self._decimal_separator.sub(r"\1\2", pred)
            numbers = self._extract_numbers.findall(pred)
            if numbers:
                new_predictions.append(numbers[-1])
            else:
                new_predictions.append(pred)
        return new_predictions

    def format_instance(self, instance):
        instance["lang"] = self.language
        instance['short_answer'] = str(instance["answer_number"])
        instance["target"] = instance["answer"]

        return instance

    @cached_property
    def references(self):
        return [instance["short_answer"] for instance in self.evaluation_data]
