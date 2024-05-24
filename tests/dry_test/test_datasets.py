import pytest

from ..fixtures import run_evaluate

datasets = {
    "agieval": [],
    "alpaca_eval": None,
    "anli": [],
    "arc": [],
    "bbh": [],
    "boolq": [],
    "cb": [],
    "ceval": [],
    "cmmlu": [],
    "cnn_dailymail": [],
    "color_objects": [],
    "commonsenseqa": [],
    "copa": [],
    "coqa": None,
    "crows_pairs": None,
    "drop": [],
    "gaokao": [],
    "gsm8k": [],
    "gpqa": [],
    "halueval": [],
    "hellaswag": [],
    "humaneval": ["--pass_at_k", "1"],
    "ifeval": [],
    "lambada": [],
    "math": [],
    "mbpp": ["--pass_at_k", "1"],
    "mmlu": [],
    "mrpc": [],
    "mt_bench": None,
    "nq": [],
    "openbookqa": [],
    "penguins_in_a_table": [],
    "piqa": [],
    "qnli": [],
    "quac": [],
    "race": [],
    "real_toxicity_prompts": None,
    "rte": [],
    "siqa": [],
    "squad": [],
    "squad_v2": [],
    "story_cloze": None,
    "tldr": [],
    "triviaqa": [],
    "truthfulqa_mc": [],
    "tydiqa": [],
    "vicuna_bench": None,
    "webq": [],
    "wic": [],
    "winogender": [],
    "winograd": [],
    "winogrande": [],
    "wmt16:de-en": [],
    "wsc": [],
    "xcopa": [],
    "xlsum": [],
    "xsum": [],
}


@pytest.mark.parametrize("dataset, extra_args", datasets.items())
def test_datasets_dry_run(run_evaluate, dataset, extra_args):
    if extra_args is None:
        return
    run_evaluate(
        ["-m", "gpt-3.5-turbo", "-d", dataset, "-b", "10", "--dry_run", "--cuda", "0", "--openai_api_key", "fake_key"] +
        extra_args
    )


def test_crows_pairs_dry_run(run_evaluate):
    run_evaluate(["-m", "gpt2", "-d", "crows_pairs", "-b", "10", "--dry_run", "--cuda", "0"])
