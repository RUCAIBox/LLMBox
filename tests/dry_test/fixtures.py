import os
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import requests


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want

    clear = os.environ.get("GITHUB_ACTION", None) == "1"
    path = Path.home() / ".cache/huggingface/datasets"

    if clear:
        path.mkdir()

    yield  # this is where the testing happens

    if clear:
        path.rmdir()


@pytest.fixture
def run_evaluate():

    def evaluate(args: List[str], cuda: str = "", test_evaluation_data: Optional[Dict[str, str]] = None):
        if cuda:
            if isinstance(cuda, int):
                cuda = str(cuda)
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda

        from datasets.exceptions import DatasetGenerationError

        from utilization import get_evaluator, parse_argument

        model_args, dataset_args, evaluation_args = parse_argument(
            args=args,
            initalize=True,
        )

        try:
            evaluator = get_evaluator(
                model_args=model_args,
                dataset_args=dataset_args,
                evaluation_args=evaluation_args,
                initalize=False,
            )
            evaluator.evaluate()
        except (ConnectionError, requests.exceptions.ReadTimeout):
            pytest.skip(reason="ConnectionError")
        except DatasetGenerationError:
            pytest.skip(reason="DatasetGenerationError")

        if test_evaluation_data is not None:
            for key, value in test_evaluation_data.items():
                if key in evaluator.dataset._datasets:
                    assert evaluator.dataset._datasets[key].evaluation_data[0] == value

    return evaluate
