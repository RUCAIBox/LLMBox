import json
import typing
from collections import OrderedDict
from copy import copy
from logging import getLogger
from pprint import pformat
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm as tqdm_lib

from .enum import GAOKAO_CHINESE_TASKS, GAOKAO_ENGLISH_TASKS
from .enum import AGIEVAL_ENGLISH_TASK, AGIEVAL_CHINESE_TASK, AGIEVAL_GAOKAO_TASK
from .icl_strategies import ape, global_entropy_ordering_strategy, knn_construct_examples
from .utils import get_raw_dataset_loader

if typing.TYPE_CHECKING:
    # solve the circular import
    from ..metric.metric import Metric
    from ..model.model import Model
    from ..utils import DatasetArguments

logger = getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    r"""The base class representing a dataset for a specific task.

    Class Attributes:
        - `name (str)`: The name of this dataset.
        - `instruction (str)`: Dataset-specific instruction for this task.
        - `metrics (List)`: The metric functions used for evaluation.
        - `evaluation_type (Literal['ranking', 'generation', 'user-defined'])`: The type of evaluation for the dataset.
        - `evaluation_set (str)`: The evaluation split of the dataset. Evaluation data will be automatically loaded.
        - `example_set (Optional[str])`: The example split of the dataset. Example data will be automatically loaded if this is not None.
        - `load_args (Union[Tuple[str], Tuple[str, str], Tuple[()]])`: Arguments for loading the dataset with huggingface `load_dataset`.

    Attributes:
        - `args (DatasetArguments)`: The arguments for the dataset.
        - `model (Model)`: The model used for the dataset.
        - `tokenizer (Tokenizer)`: The tokenizer used for the dataset.
        - `num_shots (int)`: The number of few-shot examples to construct.
        - `max_example_tokens (int)`: The maximum number of tokens allowed for the few-shot examples.
        - `examples (str)`: The constructed demonstration text.
        - `evaluation_data (List[Dict])`: The loaded evaluation data.
        - `example_data (List[Dict])`: The loaded example data.
        - `evaluation_instances (List[str])`: The final formatted instances for evaluation.
        - `option_nums (List[int])`: The number of options for each evaluation instance.
    """

    instruction: str
    r"""Dataset-specific instruction for the task."""

    metrics: List["Metric"]
    r"""The metric functions used for evaluation."""

    evaluation_type: Literal["ranking", "generation", "user-defined"]
    r"""The type of evaluation for the dataset."""

    evaluation_set: str
    r"""The evaluation split of dataset. Evaluation data will be automatically loaded."""

    example_set: Optional[str]
    r"""The example split of dataset. Example data will be automatically loaded if this is not None."""

    load_args: Union[Tuple[str], Tuple[str, str], Tuple[()]]
    r"""Arguments for loading the dataset with huggingface `load_dataset`.

    Supported formats:
        - `(dataset_name,)`: If the dataset only has one subset. E.g., `('race',)`. Or the dataset has more than one subset name. E.g., `("allenai/ai2_arc",)` accepts command line argument `--dataset arc:ARC-Easy,ARC-Challenge`.
        - `(dataset_name, subset_name)`: If the dataset is a subset of a dataset collection. E.g., `('super_glue', 'copa')`.
        - `()`: Sepcial case like `wmt` dataset.
    """

    extra_model_args: Dict[str, typing.Any] = dict()
    """Arguments for the model generation or get_ppl. See `set_generation_args` or `set_ppl_args` for details."""

    category_column: Optional[str] = None
    """The column name of the categories, e.g., winogender. Used to calculate the metric for each category."""

    category_subsets: Optional[Dict[str, List[str]]] = None
    """The subsets of each category, e.g., mmlu. Used to calculate the metric for each category."""

    _repr = [
        "name",
        "subset_name",
        "instruction",
        "metrics",
        "evaluation_type",
        "model_evaluation_method",
        "evaluation_set",
        "example_set",
        "load_args",
        "extra_model_args",
    ]

    def __init__(self, args: "DatasetArguments", model: "Model", subset_name: Optional[str] = None):
        r"""This should be called by the subclass.

        Args:
            - `args (DatasetArguments)`: The arguments for the dataset.
            - `model (Model)`: Our class for model.
            - `subset_name (Optional[str])`: The subset name of the dataset. Used when loading raw dataset from huggingface.
        """
        super().__init__()
        self.args = args
        self.name = args.dataset_name
        self.subset_name = subset_name
        self.model = model
        self.tokenizer = model.tokenizer

        self._post_init_arguments()

        self.num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.examples = ""
        self.kate = args.kate
        self.globale = args.globale
        self.ape = args.ape

        # load `self.evaluation_data` and `self.example_data`
        self.load_raw_dataset(
            dataset_path=args.dataset_path,
            subset_name=subset_name,
            evaluation_set=args.evaluation_set or self.evaluation_set,
            example_set=args.example_set or self.example_set,
        )

        if self.num_shots:
            if self.ape or self.kate or self.globale:
                self.formatted_example_data = [
                    self._format_instance(data, format_example=True) for data in self.example_data
                ]
            if len(self.example_data) < self.num_shots:
                logger.warning(
                    f"The example data only has {len(self.example_data)} instances, but the few-shot number is set to {self.num_shots}. Setting the few-shot number to {len(self.example_data)}."
                )
                self.num_shots = len(self.example_data)
            if len(self.example_data) == self.num_shots:
                self.random_indice = list(range(len(self.example_data)))
            else:
                self.random_indice = np.random.choice(len(self.example_data), self.num_shots, replace=False)
        self.formatted_evaluation_data = [self._format_instance(data) for data in self.evaluation_data]
        self.construct_instances()
        logger.debug(self)

    def _post_init_arguments(self):

        extra_model_args = copy(self.extra_model_args)

        # sample num
        if self.args.sample_num > 1 and self.model_evaluation_method in {"get_ppl", "get_prob"}:
            self.args.sample_num = 1
            logger.warning(
                f"Self-consistency only supports evaluation using the generation mode, automatically set sample_num = 1."
            )

        # temperature
        if "temperature" in self.extra_model_args and self.model.args.temperature is None:
            self.model.args.temperature = self.extra_model_args["temperature"]
        if self.args.sample_num > 1 and self.model_evaluation_method == "generation" and self.model.args.temperature == 0:
            self.model.args.temperature = 1
            logger.warning(
                f"Self-consistency only supports generation with temperature > 0, automatically set temperature = 1."
            )

        # ranking with options
        if self.name == "winogrande" and self.args.ranking_with_options:
            logger.warning(
                f"Winogrande does not support ranking with options, automatically set ranking_with_options = False."
            )
            self.args.ranking_with_options = False

        # add stop for generatino
        if isinstance(self.model.args.stop, str) and len(self.model.args.stop) > 0:
            cmd_stop = [self.model.args.stop]
        elif isinstance(self.model.args.stop, list) and len(self.model.args.stop) > 0:
            cmd_stop = self.model.args.stop
        else:
            cmd_stop = []
        if len(cmd_stop) > 0:
            extra_model_args["stop"] = extra_model_args.get("stop", []) + cmd_stop

        if self.model_evaluation_method == "get_ppl":
            self.model.set_ppl_args(**extra_model_args)
        elif self.model_evaluation_method == "generation":
            self.model.set_generation_args(**extra_model_args)
        elif self.model_evaluation_method == "get_prob":
            self.model.set_prob_args(**extra_model_args)

        logger.info(self.model.args)
        logger.info(self.args)

    @property
    def model_evaluation_method(self) -> Literal['get_ppl', 'get_prob', 'generation', 'user_defined']:
        if not hasattr(self, "args"):
            raise ValueError("The `args` attribute is not found. Please call `__init__` first.")
        if self.evaluation_type == "ranking":
            if self.args.ranking_type.startswith("ppl"):  # ppl or ppl_no_option
                return "get_ppl"
            elif self.args.ranking_type == "prob":
                return "get_prob"
        elif self.evaluation_type == "generation":
            return "generation"
        elif self.evaluation_type == "user_defined":
            return "user_defined"
        else:
            raise ValueError(
                f"We only support three evaluation types: `ranking`, `generation`, and `user_defined`, but got `{self.evaluation_type}`."
            )

    def load_raw_dataset(
        self,
        dataset_path: str,
        subset_name: Optional[str],
        evaluation_set: str,
        example_set: Optional[str],
    ):
        r"""Load the raw dataset from huggingface or local path into `self.evaluation_data` and `self.example_data`.

        Load from huggingface server:
        ```bash
        python inference.py --dataset copa
        python inference.py --dataset race:middle,high
        python inference.py --dataset race:middle,high --evaluation_set test --example_set train
        ```

        ---

        If `dataset_path` is not None, the dataset will be loaded from the given local path:

        ```bash
        # from a cloned directory of the huggingface dataset repository:
        python inference.py --dataset copa --dataset_path /path/to/copa

        # from a local (nested) directory saved by `dataset.save_to_disk`:
        python inference.py --dataset race --dataset_path /path/to/race/middle
        python inference.py --dataset race:middle --dataset_path /path/to/race
        python inference.py --dataset race:middle --dataset_path /path/to/race/middle
        python inference.py --dataset race:middle,high --dataset_path /path/to/race
        ```

        `dataset_path` can also accept a dataset file or a directory containing these files (supports json, jsonl, csv, and txt):
        ```bash
        # load one split from one subset only
        python inference.py --dataset gsm8k --dataset_path /path/to/gsm.jsonl
        python inference.py --dataset race --dataset_path /path/to/race/middle/train.json

        # load test and train splits from middle subset (a directory contains `/path/to/race/middle/train.json` and `/path/to/race/middle/test.json`)
        python inference.py --dataset race --dataset_path /path/to/race/middle --evaluation_set test --example_set train

        # load test and train splits from middle and high subsets (a nested directory)
        python inference.py --dataset race:middle,high --dataset_path /path/to/race --evaluation_set test --example_set train

        # load test and train splits from middle and high subsets with a filename pattern
        python inference.py --dataset race:middle,high --evaluation_set test --example_set train --dataset_path "/pattern/of/race_{subset}_{split}.json"
        python inference.py --dataset mmlu --evaluation_set val --example_set dev --dataset_path "/pattern/of/mmlu/{split}/{subset}_{split}.csv"
        ```

        ---

        Also feel free to override this function if you want to load the dataset in a different way:

        ```python
        from .utils import load_raw_dataset_from_file, get_raw_dataset_loader

        class MyDataset(Dataset):
            def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
                self.evaluation_data = get_raw_dataset_loader(...)("test")
                self.example_data = load_raw_dataset_from_file("examples.json")
        ```
        """

        load_fn, msg = get_raw_dataset_loader(
            dataset_name=self.name,
            dataset_path=dataset_path,
            subset_name=subset_name,
            load_args=self.load_args,
            return_msg=True,
        )  # type: ignore
        logger.info(
            msg + f" with evaluation set `{evaluation_set}`" +
            (f" and example set `{example_set}`" if example_set else "")
        )

        self.evaluation_data = list(load_fn(evaluation_set))
        if self.num_shots:
            self.example_data = list(load_fn(example_set)) if example_set else []

        logger.info(f"Evaluation data with {len(self.evaluation_data)} instances")
        logger.info(f"The example instance:\n{pformat(self.evaluation_data[0], sort_dicts=False)}")

    def __len__(self):
        return len(self.evaluation_instances)

    def __getitem__(self, idx):
        return self.evaluation_instances[idx]

    @property
    def references(self):
        r"""Get the references for `evaluation_data`.

        Returns:
            List[str]: The list of ground-truth answers.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `references` property.")

    def construct_instances(self):
        r"""Construct and format all the instances of `evaluation_data`.

        Returns:
            List[str]: The list of final formatted instances.
        """
        # automatic instruction
        if self.ape is True:
            instrction = ape(
                self.formatted_example_data, self.formatted_evaluation_dataset, self.model.get_ppl, self.model.api_key
            )
            self.instruction = instrction

        self.evaluation_instances = []
        self.option_nums = []
        if self.model_evaluation_method == "get_ppl":
            for formatted_instance in self.formatted_evaluation_data:
                instance_with_examples = self.format_instruction_and_examples(formatted_instance)
                if "options" in formatted_instance:
                    options = [(instance_with_examples, option) for option in formatted_instance["options"]]
                    self.evaluation_instances.extend(options)
                    self.option_nums.append(len(options))
                else:
                    # multiple contexts instead of options, cases like winogrande
                    contexts = [(context, formatted_instance["target"]) for context in instance_with_examples]
                    self.evaluation_instances.extend(contexts)
                    self.option_nums.append(len(contexts))
            logger.info("Evaluation mode: calculate PPL of the optional text based on the source text")
            logger.info("Formatted example (source)\n" + pformat(self.evaluation_instances[0][0]))
            logger.info(f"Formatted example (option)\n" + pformat(self.evaluation_instances[0][1]))
            if len(self.evaluation_instances) > 1:
                logger.debug("Next formatted example (source)\n" + pformat(self.evaluation_instances[1][0]))
                logger.debug("Next formatted example (option)\n" + pformat(self.evaluation_instances[1][1]))

        elif self.model_evaluation_method == "generation":
            for formatted_instance in self.formatted_evaluation_data:
                self.evaluation_instances.append(self.format_instruction_and_examples(formatted_instance))
            logger.info("Evaluation mode: generation based on the source text")
            logger.info("Formatted example (source)\n" + self.evaluation_instances[0])
            if len(self.evaluation_instances) > 1:
                logger.debug("Next formatted example (source)\n" + self.evaluation_instances[1])

        elif self.model_evaluation_method == "get_prob":
            for formatted_instance in self.formatted_evaluation_data:
                self.evaluation_instances.append(self.format_instruction_and_examples(formatted_instance))
                self.option_nums.append(len(formatted_instance["options"]))
            self.evaluation_instances = list(zip(self.evaluation_instances, self.option_nums))
            logger.info("Evaluation mode: get the probability of each option label")
            logger.info("Formatted example (source)\n" + pformat(self.evaluation_instances[0][0]))
            if len(self.evaluation_instances) > 1:
                logger.debug("Next formatted example (source)\n" + pformat(self.evaluation_instances[1][0]))
            logger.debug(f"option_nums: {self.option_nums}")

        else:
            for formatted_instance in self.formatted_evaluation_data:
                self.evaluation_instances.append(self.format_instruction_and_examples(formatted_instance))
            logger.info("Evaluation mode: user defined")
            logger.info("Formatted example (source)\n" + pformat(self.evaluation_instances[0]))
            if len(self.evaluation_instances) > 1:
                logger.debug("Formatted example (source)\n" + pformat(self.evaluation_instances[1]))

        # for self-consistency
        self.evaluation_instances = self.evaluation_instances * self.args.sample_num
        self.option_nums = self.option_nums * self.args.sample_num

    def _format_instance(self, instance, loose: bool = False, format_example: bool = False):
        """Format the dataset instance into task source text, target text, and options (for ranking).

        Args:
            `instance (Dict)`: an instance dict of multiple key-value pairs.
            `loose (bool, optional)`: Whether to add extra newline characters. Defaults to False.
            `format_example (bool, optional):` Whether to format the example. This will only effect datasets like winogrande by returning the correct source only. Defaults to False.
        """
        # it is not recommended to modify instance, in case of multiple calls
        formatted_instance = self.format_instance(instance)
        loose = "\n" if loose else ""

        if self.evaluation_type == "ranking" and "target_idx" in formatted_instance:
            if self.args.ranking_with_options:
                # update options with labels and then append options to source
                for i, option in enumerate(formatted_instance["options"]):
                    formatted_instance["options"][i] = chr(65 + i) + ". " + option.lstrip()
                formatted_instance["source"] += "\n" + loose + "\n".join(formatted_instance["options"]) + loose
                # loose: "[source]\n\n[options]\n[source_postfix]"
                # not loose: "[source]\n[options]\n[source_postfix]"

            target_idx = formatted_instance.pop("target_idx")
            if self.model_evaluation_method == "get_ppl":
                formatted_instance["target"] = formatted_instance["options"][target_idx]
            elif self.model_evaluation_method == "get_prob":
                formatted_instance["target"] = chr(65 + target_idx)

        if "source_postfix" in formatted_instance:
            formatted_instance["source"] += formatted_instance.pop("source_postfix")

        if not formatted_instance["target"].startswith(" "):
            formatted_instance["target"] = " " + formatted_instance["target"]

        if format_example and "source_idx" in formatted_instance:
            formatted_instance["source"] = formatted_instance["source"][formatted_instance.pop("source_idx")]

        return formatted_instance

    def format_instance(self, instance: dict) -> dict:
        r"""Format the dataset instance into task source text, target text, and options (for ranking).

        Notes:
            The instance should not be mutated since the function might be called for multiple times when formatting examples.

        Args:
            instance (Dict): an instance dict of multiple key-value pairs.

        Returns:
            `Dict`: A dictionary containing the formatted instance.
                `source` (`Unino[str, List[str]]`): The source text. If this is a list, `source_idx` is required.
                `source_idx` (`int`, optional): The index of the correct source (for multiple contexts ranking dataset like winogrande).
                `source_postfix` (`str`, optional): The postfix of the source text. This will be appended to the source text after options when `ranking_with_options` is True.
                `target` (`str`, optional): The target text. Either `target` or `target_idx` should be provided.
                `target_idx` (`int`, optional): The index of the target in the options (for ranking). This will generate the `target` text in `_format_instance`.
                `options` (`List[str]`, optional): The options for ranking.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `format_instance` function.")

    def format_instruction_and_examples(self, instance) -> Union[str, List[str]]:
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (dict): the pre-formatted source.

        Returns:
            Union[str, List[str]]: The final formatted instance. Return a list of formatted instances if the source is a list (in cases like winogrande).
        """
        if self.examples == "" or self.kate or self.globale:
            self.examples = self.construct_examples(instance)

        if self.model.type not in ["base", "instruction"]:
            raise ValueError(
                f"Invalid model type: {self.model.type}. Please use `--model_type` to specify the"
                " model type, which can be chosen from `base` and `instruction`."
            )

        if isinstance(instance["source"], list):
            # return a list of formatted instances if the source is a list (in cases like winogrande)
            sources = [
                self.examples + self.args.instance_format.format(source=s, target="") for s in instance["source"]
            ]
            if self.model.type == "instruction":
                sources = [self.instruction + "\n\n" + s for s in sources]

            return sources
        else:
            source = self.examples + self.args.instance_format.format(source=instance["source"], target="")
            if self.model.type == "instruction":
                source = self.instruction + "\n\n" + source

            return source

    def construct_examples(self, instance=None) -> str:
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Dict): a pre-formatted evaluation instance.

        Returns:
            str: The constructed demonstration text.
        """
        if self.num_shots == 0:
            return ""
        elif len(self.example_data) == 0:
            raise ValueError(
                f"Receive num_shots={self.num_shots}, but cannot construct examples for dataset {self.name} without example data."
            )

        if self.kate is True:
            if isinstance(instance["source"], list):
                instance_source = instance["source"][instance["source_idx"]]
            else:
                instance_source = instance["source"]

            # select demonstrations based on knn algorithm
            # TODO: Bugs in kate, order, filter
            indice = knn_construct_examples(instance_source, self.formatted_example_data, self.num_shots)
        else:
            indice = self.random_indice

        if self.globale is True:
            # rank demonstrations based on global entropy
            labels = list(range(len(self.formatted_example_data[0]["options"])))
            indice = global_entropy_ordering_strategy(indice, labels, self.formatted_example_data, self.model.get_ppl)

        # construct few-shot examples
        example_text = ""
        example_token_nums = 0
        for index in indice:
            if hasattr(self, "formatted_example_data"):
                example = self.formatted_example_data[index]
            else:
                example = self._format_instance(self.example_data[index], format_example=True)
            cur_example_text = self.args.instance_format.format_map(example) + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + example_token_nums <= self.max_example_tokens:
                example_text += cur_example_text
                example_token_nums += cur_token_num

        return example_text

    def calculate_metric(self, predictions) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]:
        r"""Calculate the metric score between `predictions` and `references`.

        Args:
            predictions (List[Union[int, str]]): The predicted answers.

        Returns:
            Dict[str, Dict[str, float]]: The metric results in the format `{"Dataset Name": {"Metric": Score}}`.
            Dict[str, List[float]]: The score lists.
        """

        def _calculate_metric(predictions, references):
            results = {}
            for metric_func in self.metrics:
                results.update(metric_func(predictions, references))
            return results

        score_lists = {}
        overall_results = _calculate_metric(predictions, self.references)
        for metric_func in self.metrics:
            score_lists.update(metric_func.last_score_lists())

        subject_results = {}
        if self.category_column is not None:
            subject_results = pd.DataFrame({
                "predictions": predictions,
                "references": self.references,
                "subject": map(lambda i: f"{self.name}[{i[self.category_column]}]", self.evaluation_data),
            }).groupby("subject").apply(lambda df: _calculate_metric(df["predictions"], df["references"])).to_dict()

        metric_results = OrderedDict(**subject_results)
        metric_results[self.name + (f":{self.subset_name}" if self.subset_name else "")] = overall_results
        return metric_results, score_lists

    def post_processing(self, predictions: Union[List[Tuple[str, float]], List[List[float]]]):
        r"""Post processing for the predictions.

        Args:
            predictions (List[Union[str, float]]): The predicted answers (generated texts or perplexity scores).

        Returns:
            List[Union[str, float]]: The post-processed predictions.
        """
        return predictions

    def log_predictions(
        self,
        raw_predictions: List[str],
        processed_predictions: Optional[List[Union[str, float]]] = None,
        score_lists: Optional[Dict[str, List[float]]] = None,
        file: Optional[str] = None,
        _to_json: bool = True,
    ) -> pd.DataFrame:
        r"""Save the dataset inputs and corresponding model predictions to file. Log intermediate results with `log_predictions(raw_predictions)` and log final results with `log_predictions(raw_predictions, processed_predictions, score_lists)`.

        Args:
            raw_predictions (List[str]): The raw predictions of model.
            processed_predictions (Optional[List[Union[str, float]]]): The processed answers.
            file (Optional[str]): The file path to save the predictions. If None, will use `args.evaluation_results_path`.
        """

        file = file or self.args.evaluation_results_path
        if processed_predictions is not None:
            assert score_lists is not None and score_lists is not None
            transposed_score_lists = [dict(zip(score_lists.keys(), values)) for values in zip(*score_lists.values())]

        def repeat_iter(obj, n):
            for _ in range(n):
                yield from obj

        def to_dict(merge: Optional[List[str]] = None, merge_by_option: Optional[List[str]] = None):
            merge = merge or []
            merge_by_option = merge_by_option or []

            def wrapper(df):
                df_dict = df.to_dict(orient="list")
                for col in merge:
                    df_dict[col] = df_dict[col][0]
                if "option_num" in df_dict:
                    option_num = df_dict.pop("option_num")[0]
                    for col in merge_by_option:
                        df_dict[col] = df_dict[col][:option_num]
                return df_dict

            return wrapper

        if processed_predictions is None:
            # log intermediate results only
            if not hasattr(self, "_lines_iter"):
                self._lines_iter = zip(
                    range(self.len()), self.evaluation_instances, repeat_iter(self.references, self.args.sample_num)
                )
            for idx, source, reference in self._lines_iter:
                lines = {
                    "index": idx,
                    "source": source,
                    "raw_prediction": raw_predictions[-1],
                    "reference": reference,
                }
                if _to_json:
                    try:
                        with open(file, "a") as f:
                            json.dump(lines, f, ensure_ascii=False)
                            f.write("\n")
                    except Exception as e:
                        logger.warning(f"Failed to log_predictions: {e}\n{lines}")
                return lines
            return None

        elif self.model_evaluation_method == "generation":
            # only generation tasks support self-consistency
            lines = {
                "index": repeat_iter(range(len(self.evaluation_data)), self.args.sample_num),
                "source": self.evaluation_instances,
                "raw_prediction": raw_predictions,
                "processed_prediction": processed_predictions,
                "reference": repeat_iter(self.references, self.args.sample_num),
                "metric": repeat_iter(transposed_score_lists, self.args.sample_num),
            }
            try:
                lines = pd.DataFrame(lines).groupby("index").apply(
                    to_dict(merge=["index", "source", "metric", "reference"])
                )
                if _to_json:
                    lines.to_json(file, orient="records", indent=4, force_ascii=False)
                return lines
            except Exception as e:
                lines = {k: len(v) for k, v in lines.items()}
                logger.warning(f"Failed to log_predictions: {e}\n{lines}")
                return None

        elif self.model_evaluation_method == "get_ppl":  # ranking

            def repeat_by_option(*arr):

                def wrapper():
                    for cols in zip(range(len(self.option_nums)), *arr):
                        for _ in range(self.option_nums[cols[0]]):
                            yield (*cols, self.option_nums[cols[0]])

                return zip(*wrapper())

            source_text, target_text = zip(*self.evaluation_instances)
            if self.use_normalization:
                source_text, target_text, raw_predictions = source_text[::2], target_text[::2], raw_predictions[::2]
            index, references, transposed_score_lists, option_nums = repeat_by_option(
                self.references, transposed_score_lists
            )
            lines = {
                "index": index,
                "source": source_text,
                "option": target_text,
                "option_num": option_nums,
                "perplexity": map(lambda r: r[0], raw_predictions),
                "reference": references,
                "metric": transposed_score_lists,
            }
            try:
                if self.name == "winogrande":
                    merge = ["index", "option", "reference", "metric"]
                    merge_by_option = ["source"]
                else:
                    merge = ["index", "source", "reference", "metric"]
                    merge_by_option = ["option"]
                lines = pd.DataFrame(lines).groupby("index").apply(to_dict(merge, merge_by_option))
                if _to_json:
                    lines.to_json(file, orient="records", indent=4, force_ascii=False)
                return lines
            except Exception as e:
                lines = {k: len(v) for k, v in lines.items()}
                logger.warning(f"Failed to log_predictions: {e}\n{lines}")
                return None

        elif self.model_evaluation_method == "get_prob":

            lines = {
                "index": range(len(self.evaluation_data)),
                "source": map(lambda i: i[0], self.evaluation_instances),
                "probabilites": raw_predictions,
                "prediction": processed_predictions,
                "reference": self.references,
                "metric": transposed_score_lists,
            }
            try:
                lines = pd.DataFrame(lines)
                if _to_json:
                    lines.to_json(file, orient="records", indent=4, force_ascii=False)
                return lines
            except Exception as e:
                lines = {k: len(v) for k, v in lines.items()}
                logger.warning(f"Failed to log_predictions: {e}\n{lines}")
                return None

        else:
            logger.debug(
                f"Failed to log predictions: processed_predictions={processed_predictions}, model_evaluation_method={self.model_evaluation_method}"
            )
            return None

    def last_score_lists(self) -> Dict[str, List[float]]:
        results = {}
        for metric in self.metrics:
            results.update(metric.last_score_lists())
        return results

    @property
    def use_normalization(self) -> bool:
        return self.name in {"arc", "openbookqa", "race"}

    def len(self, sample_num: bool = True, option_num: bool = True, normalization: bool = True) -> int:
        """Provides a unified interface to retrieve the length of dataset`.

        - `len(dataset.evaluation_data)` or `len(dataset.evaluation_data)`: the length of raw evaluation data
        - `len(dataset)` or `len(dataset.evaluation_instances)`: the length of `__iter__`. Equal to length of raw data multiplied by `args.sample_num`, option_num (if `model_evaluation_method` is "get_ppl") and 2 (if `use_normalization` is True)
        """
        # if `model_evaluation_method` is not "get_ppl", two branches of `option_num` should be equivalent
        if option_num:
            length = len(self.evaluation_instances)
            if not sample_num and self.args.sample_num > 1:
                length = length // self.args.sample_num
            if not normalization and self.use_normalization:
                length = length // 2
        else:
            length = len(self.evaluation_data)
            if sample_num and self.args.sample_num > 1:
                length *= self.args.sample_num
            if normalization and self.use_normalization:
                length *= 2
        return length

    def update_tqdm(self, tqdm):
        # do nothing
        pass

    def __repr__(self):
        reprs = [f"{p}={getattr(self, p)!r}" for p in self._repr]
        reprs.append(f"len={len(self)}")
        return "Dataset(" + ", ".join(reprs) + ")"


class DatasetCollection(torch.utils.data.Dataset):

    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__()
        self.subset_names = list(datasets.keys())
        self._datasets = list(datasets.values())
        self._cur_idx = 0
        self.args = self._datasets[0].args
        self._repr = copy(self._datasets[0]._repr)
        self.categorized_subsets = self._datasets[0].category_subsets
        for idx, prop in enumerate(self._repr):
            if prop == "subset_name":
                self._repr[idx] = "subset_names"
                break

    @property
    def name(self) -> str:
        return self._datasets[0].name

    @property
    def option_nums(self) -> List[int]:
        """If `model_evaluation_method` is "get_ppl", this returns the total number of options across all evaluation examples. Otherwise, this returns an empty list."""
        return sum([d.option_nums for d in self._datasets], [])

    def len(self, sample_num: bool = True, option_num: bool = True, normalization: bool = True) -> int:
        return sum(d.len(sample_num, option_num, normalization) for d in self._datasets)

    def __len__(self):
        return sum(len(d) for d in self._datasets)

    def _split_by_subset(
        self,
        obj: Optional[Union[list, dict]] = None,
        sample_num=True,
        option_num=True,
        normalization=True,
        strict=True,
    ) -> Iterator[Union[list, dict]]:
        st = 0
        if obj is None:
            yield from [None] * len(self._datasets)
        elif isinstance(obj, list):
            if strict:
                assert self.len(sample_num, option_num, normalization) == len(obj)
            for d in self._datasets:
                dlen = d.len(sample_num, option_num, normalization)
                if st >= len(obj):
                    return
                yield obj[st:st + dlen]
                st += dlen
        elif isinstance(obj, dict):
            assert all(len(v) == self.len(sample_num, option_num, normalization) for v in obj.values())
            for d in self._datasets:
                dlen = d.len(sample_num, option_num, normalization)
                yield {k: v[st:st + dlen] for k, v in obj.items()}
                st += dlen

    def log_predictions(
        self,
        raw_predictions: List[str],
        processed_predictions: Optional[List[Union[str, float]]] = None,
        score_lists: Optional[Dict[str, List[float]]] = None,
        file: Optional[str] = None
    ):
        lines = []
        raw = self._split_by_subset(raw_predictions, strict=processed_predictions is not None)
        processed = self._split_by_subset(processed_predictions, option_num=False, normalization=False)
        score = self._split_by_subset(score_lists, sample_num=False, option_num=False, normalization=False)

        if processed_predictions is None:
            for d, r, p, s in zip(self._datasets, raw, processed, score):
                results = d.log_predictions(r, p, s, file, True)
                if results is not None:
                    return
        else:
            for d, r, p, s in zip(self._datasets, raw, processed, score):

                def set_subset(l):
                    l["subset"] = d.subset_name

                df = d.log_predictions(r, p, s, file, False)
                df.apply(set_subset)
                lines.append(df)
            file = file or self.args.evaluation_results_path
            try:
                pd.concat(lines).to_json(file, orient="records", indent=4, force_ascii=False)
            except Exception as e:
                logger.debug(f"Failed to log predictions: {e}")

    def post_processing(self, predictions: List[Union[str, float]]):
        return sum((d.post_processing(p) for d, p in zip(self._datasets, self._split_by_subset(predictions))), [])

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise IndexError(f"Index {idx} out of range")
        self._cur_idx = 0
        while idx >= self._datasets[self._cur_idx].len():
            idx -= self._datasets[self._cur_idx].len()
            self._cur_idx += 1
        return self._datasets[self._cur_idx][idx]

    def __iter__(self):
        for self._cur_idx, d in enumerate(self._datasets):
            yield from d.__iter__()

    def __getattr__(self, attr):
        return getattr(self._datasets[self._cur_idx], attr)

    def calculate_metric(self, predictions) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]:
        results = OrderedDict()
        score_lists = dict()
        splitted = self._split_by_subset(predictions, option_num=False, normalization=False)
        for d, p in zip(self._datasets, splitted):
            subset_results, score_list = d.calculate_metric(p)
            results.update(subset_results)
            for k, v in score_list.items():
                score_lists.setdefault(k, []).extend(v)

        metric_entries = results[f"{self.name}:{self.subset_names[0]}"].keys()

        # append subcategories results if available
        if self.categorized_subsets and len(self.subset_names) == sum(len(s) for s in self.categorized_subsets.values()):
            for cat, cat_subjects in self.categorized_subsets.items():
                cat_results = [results[f"{self.name}:{subject}"] for subject in cat_subjects]
                results[f"{self.name}[{cat}]"] = {m: np.mean([r[m] for r in cat_results]) for m in metric_entries}

        if self.name == "gaokao":
            results[self.name + "[Chinese Mean]"] = {
                m: np.sum([
                    r[m] * GAOKAO_CHINESE_TASKS[k[7:]] for k, r in results.items() if k[7:] in GAOKAO_CHINESE_TASKS
                ]) / GAOKAO_CHINESE_TASKS["all"]
                for m in metric_entries
            }
            results[self.name + "[English Mean]"] = {
                m: np.sum([
                    r[m] * GAOKAO_ENGLISH_TASKS[k[7:]] for k, r in results.items() if k[7:] in GAOKAO_ENGLISH_TASKS
                ]) / GAOKAO_ENGLISH_TASKS["all"]
                for m in metric_entries
            }

        if self.name == "agieval":
            results[self.name + "[Chinese Mean]"] = {
                m: np.average([
                    r[m] for k, r in results.items() if k[8:] in AGIEVAL_CHINESE_TASK
                ])
                for m in metric_entries
            }
            results[self.name + "[English Mean]"] = {
                m: np.average([
                    r[m] for k, r in results.items() if k[8:] in AGIEVAL_ENGLISH_TASK
                ])
                for m in metric_entries
            }
            results[self.name + "[Gaokao Mean]"] = {
                m: np.average([
                    r[m] for k, r in results.items() if k[8:] in AGIEVAL_GAOKAO_TASK
                ])
                for m in metric_entries
            }

        if self.name == "agieval_single_choice":
            results[self.name + "[Chinese Mean]"] = {
                m: np.average([
                    r[m] for k, r in results.items() if k[22:] in AGIEVAL_CHINESE_TASK
                ])
                for m in metric_entries
            }
            results[self.name + "[English Mean]"] = {
                m: np.average([
                    r[m] for k, r in results.items() if k[22:] in AGIEVAL_ENGLISH_TASK
                ])
                for m in metric_entries
            }
            results[self.name + "[Gaokao Mean]"] = {
                m: np.average([
                    r[m] for k, r in results.items() if k[22:] in AGIEVAL_GAOKAO_TASK
                ])
                for m in metric_entries
            }

        results[self.name + "[Arithmetic Mean]"] = {
            m: np.mean([r[m] for k, r in results.items() if ":" in k])
            for m in metric_entries
        }

        return results, score_lists

    def update_tqdm(self, tqdm):
        if isinstance(tqdm, tqdm_lib.tqdm):
            tqdm.set_description(self.name + ":" + self.subset_names[self._cur_idx])

    def __repr__(self):
        reprs = [f"{p}={getattr(self, p)!r}" for p in self._repr]
        reprs.append(f"len={len(self)}")
        return "DatasetCollection(" + ", ".join(reprs) + ")"
