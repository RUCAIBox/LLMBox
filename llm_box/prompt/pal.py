from .method_oriented import get_prompt
from .base import Base
import threading
from typing import List, Optional


class PAL(Base):
    """
    A class designed to implement the 'PAL' prompting strategy in AI models.

    Large Language Model solves reasoning problems that involve complex arithmetic and procedural tasks by generating reasoning chains of text and code. This offloads the execution of the code to a program runtime, in our case, a Python interpreter.

    Paper Link: https://arxiv.org/pdf/2211.10435.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.answer_expr = "solution()"
        self.examples = get_prompt(['pal', self.dataset_name]) + '\n'

    def execute(self, code: Optional[List[str]] = None):
        exec('\n'.join(code))
        return eval(self.answer_expr)

    def post_processing(self, predictions):
        new_predictions = []
        for gens in predictions:
            if '```python' in gens:
                gens = gens.split('```python')[1].split('```')[0]
            elif '```' in gens:
                gens = gens.split('```')[1].split('```')[0]
            code = gens.split('\n')

            with Timeout():
                try:
                    exec_result = self.execute(code)
                    new_predictions.append(exec_result)
                except Exception as e:
                    new_predictions.append('')
        new_predictions = [str(x)[:-2] if str(x).endswith(".0") else str(x) for x in new_predictions]
        return new_predictions


class Timeout:
    def __init__(self, seconds=10, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
        self.timer = threading.Timer(self.seconds, self.timeout_handler)

    def timeout_handler(self):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        self.timer.start()

    def __exit__(self, type, value, traceback):
        self.timer.cancel()
