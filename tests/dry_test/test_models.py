import pytest
import torch

from .fixtures import *

models = {
    "gpt-3.5-turbo": ["--openai_api_key", "fake_key"],
    "claude-3-haiku-20240307": ["--anthropic_api_key", "fake_key"],
    "qwen-turbo": ["--dashscope_api_key", "fake_key"],
    "ERNIE-Speed": ["--qianfan_access_key", "fake_key", "--qianfan_secret_key", "fake_key"],
    "gpt2": ["--vllm", "False", "--prefix_caching", "True", "--cuda", "0"],
    "gpt2": ["--vllm", "True", "--prefix_caching", "False", "--cuda", "0"],
}


# 3 (datasets) by 6 (models) grid
@pytest.mark.parametrize("dataset", ["gsm8k", "hellaswag", "mmlu"])
@pytest.mark.parametrize("model, extra_args", models.items())
def test_models_dry_run(run_evaluate, model, dataset, extra_args):
    if not torch.cuda.is_available() and extra_args[-2:] == ["--cuda", "0"]:
        pytest.skip("CUDA is not available")

    if extra_args is None:
        return
    try:
        run_evaluate(["-m", model, "-d", dataset, "-b", "10", "--dry_run"] + extra_args, cuda=0)
    except torch.cuda.OutOfMemoryError:
        pytest.skip(f"Out of memory error on {model} {dataset}")
    except FileNotFoundError:
        pytest.skip(f"File not found error on {model} {dataset}")
