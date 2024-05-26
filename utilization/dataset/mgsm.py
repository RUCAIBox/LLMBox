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

    instruction = "Answer the following question.\n\nQuestion: {{question.replace('\n', ' ')}}\nAnswer:"

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

    def post_processing(self, predictions):
        new_predictions = []
        for pred in predictions:
            if self.args.cot == 'pal':
                if '```python' in pred:
                    pred = pred.split('```python')[1].split('```')[0]
                elif '```' in pred:
                    pred = pred.split('```')[1].split('```')[0]
                code = pred.split('\n')

                with Timeout():
                    try:
                        exec('\n'.join(code))
                        ans = eval("solution()")
                        ans = str(ans)[:-2] if str(ans).endswith(".0") else str(ans)
                        new_predictions.append(ans)
                    except:
                        new_predictions.append('')
            else:
                # replace numbers like `x,xxx` with `xxxx`
                pred = self._decimal_separator.sub(r"\1\2", pred)
                numbers = self._extract_numbers.findall(pred)
                if numbers:
                    new_predictions.append(numbers[-1])
                else:
                    new_predictions.append(pred)
        return new_predictions

    def format_instance(self, instance):
        instance['short_answer'] = str(instance["answer_number"])
        instance["target"] = instance["answer"]

        return instance

    @cached_property
    def references(self):
        return [instance["short_answer"] for instance in self.evaluation_data]


class Timeout:

    def __init__(self, seconds=10, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
