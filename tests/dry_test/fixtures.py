import os
from typing import List

import pytest


@pytest.fixture
def run_evaluate():

    def evaluate(args: List[str], cuda: str = ""):
        if cuda:
            if isinstance(cuda, int):
                cuda = str(cuda)
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda

        from utilization import Evaluator, parse_argument

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
