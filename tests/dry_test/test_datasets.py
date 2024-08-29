import nltk
import pytest
from utilization.utils.logging import list_datasets

from .fixtures import *

nltk.download('punkt')
nltk.download('punkt_tab')

datasets = {
    "agieval": [],
    "alpaca_eval": ["--inference_only", "--openai_api_key", "fake-key"],
    "anli": [],
    "arc": [],
    "bbh": [],
    "boolq": [],
    "cb": [],
    "ceval": ["--no_dataset_threading"],  # dataset threading has issues with pytest
    "cmmlu": ["--no_dataset_threading"],
    "cnn_dailymail": [],
    "color_objects": [],
    "commonsenseqa": [],
    "copa": [],
    "coqa": "skip",
    "crows_pairs": "does not support api model",
    "drop": [],
    "gaokao": [],
    "gsm8k": [],
    "gpqa": "requires authentication",
    "halueval": [],
    "hellaswag": [],
    "humaneval": ["--pass_at_k", "1"],
    "ifeval": [],
    "lambada": [],
    "math": ["--no_dataset_threading"],
    "mbpp": ["--pass_at_k", "1"],
    "mmlu": [],
    "mrpc": [],
    "mt_bench": ["--inference_only", "--openai_api_key", "fake-key"],
    "nq": [],
    "openbookqa": [],
    "penguins_in_a_table": [],
    "piqa": [],
    "qnli": [],
    "quac": [],
    "race": [],
    "real_toxicity_prompts": "skip",
    "rte": [],
    "siqa": [],
    "sst2": [],
    "squad": [],
    "squad_v2": [],
    "story_cloze": "skip",
    "tldr": [],
    "triviaqa": [],
    "truthfulqa_mc": [],
    "tydiqa": [],
    "vicuna_bench": ["--inference_only", "--openai_api_key", "fake-key"],
    "webq": [],
    "wic": [],
    "winogender": [],
    "winograd": "does not support api model",
    "winogrande": [],
    "wmt16:de-en": [],
    "wsc": [],
    "xcopa": [],
    "xlsum": "dataset too large",
    "xsum": ["--no_dataset_threading"],
}

test_evaluation_data = {
    "agieval:aqua-rat": (
        'Q: A car is being driven, in a straight line and at a uniform speed, towards the base of a '
        'vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 '
        'minutes for the angle of elevation to change from 45° to 60°. After how much more time will this '
        'car reach the base of the tower?\n'
        'None\n'
        'Answer: Among A through E, the answer is 5(√3 + 1)'
    ),
    "agieval:gaokao-mathcloze": (
        '问题：已知 $a \\in \\mathrm{R}$, 函数 $f(x)=\\left\\{\\begin{array}{l}x^{2}-4, x>2 \\\\ |x-3|+a, x '
        '\\leq 2,\\end{array}\\right.$ 若 $f[f(\\sqrt{6})]=3$, 则 $a=(\\quad)$\n'
        '答案：'
    )
}

datasets_to_test = set(list_datasets()) - {
    "wmt10", "wmt13", "wmt14", "wmt15", "wmt16", "wmt17", "wmt18", "wmt19", "wmt21", "agieval_cot",
    "agieval_single_choice"
}
for dataset in datasets_to_test:
    if dataset not in datasets:
        datasets[dataset] = []


@pytest.mark.parametrize("dataset, extra_args", datasets.items())
def test_datasets_dry_run(run_evaluate, dataset, extra_args):
    """You may re-run one of these tests (e.g. ceval) with:

    `pytest tests/dry_test/test_datasets.py::test_datasets_dry_run[ceval-extra_args7]`
    """
    if not isinstance(extra_args, list):
        return

    run_evaluate(
        ["-m", "gpt-3.5-turbo", "-d", dataset, "-b", "10", "--dry_run", "--openai_api_key", "fake_key", "-i", "5"] +
        extra_args,
        cuda=0,
        test_evaluation_data=test_evaluation_data,
    )


def test_winograd_dry_run(run_evaluate):
    run_evaluate(
        ["-m", "gpt2", "-d", "winograd", "-b", "10", "--dry_run", "-i", "5"],
        cuda=0,
        test_evaluation_data=test_evaluation_data,
    )


def test_crows_pairs_dry_run(run_evaluate):
    run_evaluate(
        ["-m", "gpt2", "-d", "crows_pairs", "-b", "10", "--dry_run", "-i", "5"],
        cuda=0,
        test_evaluation_data=test_evaluation_data,
    )
