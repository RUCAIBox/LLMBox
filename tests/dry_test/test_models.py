import pytest

from ..fixtures import run_evaluate

models = {
    "gpt-3.5-turbo": ["--openai_api_key", "fake_key"],
    "claude-3-haiku-20240307": ["--anthropic_api_key", "fake_key"],
    "qwen-turbo": ["--dashscope_api_key", "fake_key"],
    "ERNIE-Speed": ["--qianfan_access_key", "fake_key", "--qianfan_secret_key", "fake_key"],
    "gpt2": ["--vllm", "False", "--prefix_caching", "True", "--cuda", "0"],
    "gpt2": ["--vllm", "True", "--prefix_caching", "False", "--cuda", "0"],
}


@pytest.mark.parametrize("dataset", ["gsm8k", "hellaswag", "mmlu"])
@pytest.mark.parametrize("model, extra_args", models.items())
def test_models_dry_run(run_evaluate, model, dataset, extra_args):
    if extra_args is None:
        return
    run_evaluate(["-m", model, "-d", dataset, "-b", "10", "--dry_run"] + extra_args)
