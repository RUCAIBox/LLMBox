import sys

from ..fixtures import *

sys.path.append('.')
from utilization.utils.arguments import parse_argument


def test_default_vllm():
    model_args, dataset_args, evaluation_args = parse_argument(['-m', 'a-random-fake-model', '-d', 'nq', 'quac'])
    assert model_args.model_backend == "vllm"
    assert model_args.prefix_caching is None  # vllm default is False


def test_no_prefix_caching():
    # batch size is 1, so prefix caching is not used
    model_args, dataset_args, evaluation_args = parse_argument([
        '-m', 'a-random-fake-model', '-d', 'nq', 'mmlu', '-b', '1'
    ])
    assert model_args.model_backend == "huggingface"
    assert model_args.prefix_caching is False


def test_default_prefix_caching():
    # currently vllm doesn't support returning logprob for prefix caching
    model_args, dataset_args, evaluation_args = parse_argument([
        '-m', 'a-random-fake-model', '-d', 'nq', 'mmlu', '-b', '16'
    ])
    assert model_args.model_backend == "huggingface"
    assert model_args.prefix_caching is None  # huggingface default is True


def test_default_no_efficient():
    model_args, dataset_args, evaluation_args = parse_argument([
        '-m', 'a-random-fake-model', '-d', 'nq', '--vllm', 'False', '--prefix_caching', 'False'
    ])
    assert model_args.model_backend == "huggingface"
    assert model_args.prefix_caching is False
