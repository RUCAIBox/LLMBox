import json
import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from utilization import DatasetArguments, EvaluationArguments, ModelArguments, get_evaluator

GPTEVAL_DATASETS = {"alpaca_eval", "mt_bench", "vicuna_bench"}
GREEN = "\033[92m"
CLEAR = "\033[0m"


def main(file_path: str, continue_from: str = None):
    r"""Run GPTEval metrics.

    Use case: Sometimes, you might want to split the evaluation into two parts: 1. LLM generation and 2. metrics calculation. This allows for more efficient use of GPU resources for metrics that take longer to compute, such as GPTEval. You can add the `--inference_only` flag in the command line for LLM generation, which will produce a JSON file of the evaluation results. This function performs the second step of reading the JSON file and calculating the GPTEval scores.

    Example:
    >>> ls evaluation_results | grep "alpaca_eval.*\.json" | xargs -I {} python docs/examples/run_gpt_eval.py evaluation_results/{}
    or continue from a GPTEval checkpoint:
    >>> python docs/examples/run_gpt_eval.py evaluation_results/<model>-alpaca_eval-<date>.json evaluation_results/gpt-3.5-turbo-alpaca_eval-<date>.json
    """

    assert file_path.endswith(".json"), "Please provide a JSON file."
    file_name = file_path.split("/")[-1]
    assert re.match(
        r".*-[^-]*-\dshot-\d\d\d\d_\d\d_\d\d-\d\d_\d\d_\d\d.json", file_name
    ), f"Please provide a valid JSON file {file_name}."
    model, dataset = re.match(r"(.*)-([^-]*)-\dshot-\d\d\d\d_\d\d_\d\d-\d\d_\d\d_\d\d.json", file_name).groups()
    assert dataset in GPTEVAL_DATASETS, f"Please provide a valid dataset. Available datasets: {GPTEVAL_DATASETS}"

    with open(file_path, "r") as f:
        args = json.loads(f.readline())
    assert args["evaluation_results"] == "batch", "Please provide the JSON file with batch evaluation results."

    evaluator = get_evaluator(
        model_args=ModelArguments(
            model_name_or_path=args["model_args"]["model_name_or_path"],
            model_backend="openai",  # use openai model to load faster
            api_endpoint="chat/completions",
            model_type=args["model_args"]["model_type"],
        ),
        dataset_args=DatasetArguments(dataset_names=[dataset], batch_size=1),
        evaluation_args=EvaluationArguments(
            continue_from=file_path,
            log_level="warning",
            gpteval_continue_from=continue_from,
        ),
    )
    metric_results = evaluator.evaluate()

    msg = ""
    for display_name, result in metric_results.items():
        if result is None:
            continue
        msg += f"\n##### {display_name} #####"
        for key, value in sorted(result.items(), key=lambda x: x[0]):
            msg += "\n{}: {:.2f}".format(key, value)

    print(evaluator.model_args)
    print(evaluator.dataset_args)
    print(evaluator.evaluation_args)
    print(f"{GREEN}{msg}{CLEAR}")


if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Please provide the path to the JSON file."

    file = sys.argv[1].rstrip("/")
    print(f"LLM file: {file}")

    if len(sys.argv) > 2:
        continue_from = sys.argv[2].rstrip("/")
        print(f"GPTEval file: {continue_from}")
    else:
        continue_from = None

    main(file, continue_from)
