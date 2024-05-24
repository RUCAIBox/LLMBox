from typing import List

import pytest

from utilization import Evaluator, parse_argument


@pytest.fixture
def run_evaluate():
    def evaluate(args: List[str]):
        model_args, dataset_args, evaluation_args = parse_argument(
            args=args,
            initalize=True,
        )

        evaluator = Evaluator(
            model_args=model_args,
            dataset_args=dataset_args,
            evaluation_args=evaluation_args,
            initalize=False,
        )
        return evaluator.evaluate()
    return evaluate
