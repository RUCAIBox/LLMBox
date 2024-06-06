import os
from typing import Dict, List, Optional

import pytest
import requests


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want
    base_path = os.path.expanduser(os.environ.get("TEMP_HFD_CACHE_PATH", "~/.cache/huggingface"))
    path = os.path.join(base_path, "datasets")
    clear = not os.path.exists(path)

    if clear:
        os.makedirs(path)

    yield  # this is where the testing happens

    if clear:
        os.removedirs(path)


@pytest.fixture
def run_evaluate():

    def evaluate(args: List[str], cuda: str = "", test_evaluation_data: Optional[Dict[str, str]] = None):
        if cuda:
            if isinstance(cuda, int):
                cuda = str(cuda)
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda

        base_path = os.path.expanduser(os.environ.get("TEMP_HFD_CACHE_PATH", "~/.cache/huggingface"))
        path = os.path.join(base_path, "datasets")
        args.append("--hfd_cache_path")
        args.append(path)

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

        if test_evaluation_data is not None:
            for key, value in test_evaluation_data.items():
                if key in evaluator.dataset._datasets:
                    assert evaluator.dataset._datasets[key].evaluation_data[0] == value

    return evaluate
