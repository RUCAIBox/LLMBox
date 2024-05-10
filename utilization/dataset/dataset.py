import typing
from collections import OrderedDict, defaultdict
from copy import copy
from itertools import chain
from logging import getLogger
from pprint import pformat
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from utilization.utils import dynamic_stride_tqdm

from ..metric.utils import avg_metrics
from ..utils.batch_sampler import DatasetCollectionBatchSampler
from ..utils.log_results import PredictionWriter, log_final_results, repeat_iter
from .enum import GAOKAO_CHINESE_TASKS_SCORE, GAOKAO_ENGLISH_TASKS_SCORE, GAOKAO_TASKS_SCORE
from .icl_strategies import ape, global_entropy_ordering_strategy, knn_construct_examples
from .utils import DatasetUtilMixin, get_raw_dataset_loader
from ..model.model_enum import ENDPOINT_ARGS

if typing.TYPE_CHECKING:
    # solve the circular import
    from ..metric.metric import Metric
    from ..model.model import Model
    from ..utils import DatasetArguments

_InputsWithOptionNum = Union[List[Tuple[str, int]], List[Tuple[str, str, int]], List[Tuple[str, str, str, int]]]
"""Instance format for the `get_prob` model evaluation method. The tuple contains the source text and the number of options. If prefix_caching is enabled, the source text will be segmented into prefixes."""

logger = getLogger(__name__)


class Dataset(torch.utils.data.Dataset, DatasetUtilMixin):
    r"""The base class representing a dataset for a specific task.

    Class Attributes:
        - `name (str)`: The name of this dataset.
        - `instruction (str)`: Dataset-specific instruction for this task.
        - `metrics (List)`: The metric functions used for evaluation.
        - `evaluation_type (Literal['ranking', 'generation', 'user_defined'])`: The type of evaluation for the dataset.
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

    evaluation_type: Literal["ranking", "generation", "user_defined"]
    r"""The type of evaluation for the dataset."""

    evaluation_set: str = None
    r"""The evaluation split of dataset. Evaluation data will be automatically loaded."""

    example_set: Optional[str] = None
    r"""The example split of dataset. Example data will be automatically loaded if this is not None."""

    load_args: Union[Tuple[str], Tuple[str, str], Tuple[()]] = ()
    r"""Arguments for loading the dataset with huggingface `load_dataset`.

    Supported formats:
        - `(dataset_name,)`: If the dataset only has one subset. E.g., `('race',)`. Or the dataset has more than one subset name. E.g., `("allenai/ai2_arc",)` accepts command line argument `-d arc:ARC-Easy,ARC-Challenge`.
        - `(dataset_name, subset_name)`: If the dataset is a subset of a dataset collection. E.g., `('super_glue', 'copa')`.
        - `()`: Sepcial case like `wmt` dataset.
    """

    extra_model_args: Dict[str, typing.Any] = dict()
    """Arguments for the model generation or get_ppl. See `set_generation_args` or `set_ppl_args` for details."""

    category_column: Optional[str] = None
    """The column name of the categories, e.g., winogender. Used to calculate the metric for each category."""

    categorized_subsets: Optional[Dict[str, List[str]]] = None
    """The subsets of each category, e.g., mmlu. Used to calculate the metric for each category."""

    banned_subsets: Optional[List[str]] = None

    use_normalization: bool = False

    _repr = [
        "name",
        "subset_name",
        "instruction",
        "metrics",
        "evaluation_type",
        "load_args",
        "model_evaluation_method",
        "ranking_type",
        "evaluation_set",
        "example_set",
        "extra_model_args",
        "real_num_shots",
        "real_example_tokens",
    ]

    def __init__(self, dataset_name: str, args: "DatasetArguments", model: "Model", subset_name: Optional[str] = None):
        r"""This should be called by the subclass.

        Args:
            - `args (DatasetArguments)`: The arguments for the dataset.
            - `model (Model)`: Our class for model.
            - `subset_name (Optional[str])`: The subset name of the dataset. Used when loading raw dataset from huggingface.
        """
        super().__init__()
        self.args = args
        self.name = dataset_name
        self.subset_name = subset_name
        self.model = model
        self.set_tokenizer(model.tokenizer)

        self.sample_num = args.sample_num
        self.max_num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.kate = args.kate
        self.globale = args.globale
        self.ape = args.ape
        self.ranking_type = args.ranking_type
        self.model_type = self.model.model_type
        self.prefix_caching = self.model.args.prefix_caching
        self.instance_format = "{source}{target}"
        if args.instruction:
            self.instruction = args.instruction

        self._init_arguments()

        # truncated by max_example_tokens
        self.real_num_shots = None
        self.real_example_tokens = None
        self.examples = ""

        # load `self.evaluation_data` and `self.example_data`
        self.evaluation_set = args.evaluation_set or self.evaluation_set
        self.example_set = args.example_set or self.example_set
        if self.max_num_shots:
            if not self.example_set:
                raise ValueError(
                    f"Please provide the example set for dataset {self.name} to construct few-shot examples."
                )
            if "val" in self.example_set or "test" in self.example_set:
                logger.warning(
                    f"Example set is used for constructing few-shot examples, but `{self.example_set}` seems to be an evaluation set."
                )
        self.load_raw_dataset(
            dataset_path=args.dataset_path,
            subset_name=subset_name,
            evaluation_set=self.evaluation_set,
            example_set=self.example_set,
        )
        if self.args.max_evaluation_instances:
            self.evaluation_data = self.evaluation_data[:self.args.max_evaluation_instances]

        self.evaluation_instances, self.option_nums = self.construct_instances()
        logger.debug(self)

    def __len__(self):
        return len(self.evaluation_instances)

    def __getitem__(self, idx):
        return self.evaluation_instances[idx]

    def __iter__(self):
        yield from self.evaluation_instances

    def format_instance(self, instance: dict) -> dict:
        r"""Format the dataset instance into task source text, target text, and options (for ranking).

        Notes:
            The instance should not be mutated since the function might be called for multiple times when formatting examples.

        Args:
            instance (Dict): an instance dict of multiple key-value pairs.

        Returns:
            A dictionary with the following keys:

            - `source` (`Union[str, List[str]]`): The source text. If this is a list, `source_idx` is required.
            - `source_idx` (`int`, optional): The index of the correct source (for multiple contexts ranking dataset like winogrande).
            - `source_postfix` (`str`, optional): The postfix of the source text. This will be appended to the source text after options when `ranking_with_options` is True.
            - `target` (`str`, optional): The target text. Either `target` or `target_idx` should be provided.
            - `target_idx` (`int`, optional): The index of the target in the options (for ranking). This will generate the `target` text in `_format_instance`.
            - `options` (`List[str]`, optional): The options for ranking.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `format_instance` function.")

    @property
    def references(self):
        r"""Get the references for `evaluation_data`.

        Returns:
            List[str]: The list of ground-truth answers.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `references` property.")

    @property
    def dataset_name(self) -> str:
        return self.name + (f":{self.subset_name}" if self.subset_name else "")

    @property
    def model_evaluation_method(self) -> Literal['get_ppl', 'get_prob', 'generation', 'user_defined']:
        if not hasattr(self, "args"):
            raise ValueError("The `args` attribute is not found. Please call `__init__` first.")
        if self.evaluation_type == "ranking":
            if self.ranking_type.startswith("ppl"):  # ppl or ppl_no_option
                return "get_ppl"
            elif self.ranking_type == "prob":
                return "get_prob"
            elif self.ranking_type == "generation":
                return "generation"
        elif self.evaluation_type == "generation":
            return "generation"
        elif self.evaluation_type == "user_defined":
            return "user_defined"
        else:
            raise ValueError(
                f"We only support three evaluation types: `ranking`, `generation`, and `user_defined`, but got `{self.evaluation_type}`."
            )

    @model_evaluation_method.setter
    def model_evaluation_method(self, value: str):
        if value not in ["get_ppl", "get_prob", "generation", "user_defined"]:
            raise ValueError(f"Invalid model evaluation method: {value}")
        if value in ["get_ppl", "get_prob"]:
            if self.evaluation_type != "ranking":
                raise ValueError(f"Model evaluation method {value} is only available for ranking datasets.")
            self.ranking_type = "ppl_no_option" if value == "get_ppl" else "prob"
        elif value == "generation":
            if self.evaluation_type == "ranking":
                self.ranking_type = "generation"
            else:
                self.evaluation_type = "generation"
        else:
            self.evaluation_type = "user_defined"

    def init_arguments(self):
        """Initialize the dataset attributes and extra_model_args. This is called before data formatting."""
        return

    def _init_arguments(self):
        """Initialize the dataset attributes and extra_model_args from `DatasetArguments` and `ModelArguments`.

        You should NOT modify `self.args` or `self.model.args` because multiple datasets are sharing the same arguments."""

        self.init_arguments()

        self._extra_model_args = copy(self.extra_model_args)

        if self.model.args.api_endpoint is not None:
            endpoint_args = ENDPOINT_ARGS[self.model.args.model_backend + "/" + self.model.args.api_endpoint]
            methods = ["get_ppl", "get_prob", "generation"]
            requireds = [
                ("echo", "max_tokens", "logprobs"),
                ("max_tokens", "temperature", "logit_bias"),
                ("max_tokens", "temperature"),
            ]
            support = [m for m, r in zip(methods, requireds) if all(a in endpoint_args for a in r)]
            if self.model_evaluation_method not in support:
                logger.warning(
                    f"Model {self.model.args.model_name_or_path} does not support {self.model_evaluation_method}, "
                    f"automatically switch to {support[0]}."
                )
                self.model_evaluation_method = support[0]

        # sample num
        if self.sample_num > 1 and self.model_evaluation_method in {"get_ppl", "get_prob"}:
            self.sample_num = 1
            logger.warning(
                f"Self-consistency only supports evaluation using the generation mode, automatically set sample_num = 1."
            )

        # temperature
        if self.sample_num > 1 and getattr(self._extra_model_args, "temperature", 0) == 0:
            self._extra_model_args["temperature"] = 1
            logger.warning(
                f"Self-consistency only supports generation with temperature > 0, automatically set temperature = 1."
            )

        logger.info(self.model.args)
        logger.info(self.args)

    def load_raw_dataset(
        self,
        dataset_path: Optional[str],
        subset_name: Optional[str],
        evaluation_set: str,
        example_set: Optional[str],
    ):
        r"""Load the raw dataset from huggingface or local path into `self.evaluation_data` and `self.example_data`."""

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
        self.example_data = list(load_fn(example_set)) if example_set else []

        logger.info(
            f"Evaluation data with {len(self.evaluation_data)} instances:\n" +
            pformat(self.evaluation_data[0], sort_dicts=False)
        )

    def construct_instances(self):
        r"""Construct and format all the instances of `evaluation_data`.

        1. Format the example data with `format_instance`
        2. Format the evaluation data with `format_instance`
        3. For each instance
            evaluation instance = instruction + examples + instance:
            1. Dynamically construct the instruction and examples with `construct_instruction_and_examples`
            2. Construct final `evaluation_instances` and `option_nums` based on the model evaluation method.

        Returns:
            List[str]: The list of final formatted instances.
        """
        if self.max_num_shots:
            if self.ape or self.kate or self.globale:
                self.formatted_example_data = [
                    self._format_instance(data, format_example=True) for data in self.example_data
                ]
            if len(self.example_data) < self.max_num_shots:
                logger.warning(
                    f"The example data of {self.dataset_name} only has {len(self.example_data)} instances, but the few-shot number is set to {self.max_num_shots}. Setting the few-shot number to {len(self.example_data)}."
                )
                self.max_num_shots = len(self.example_data)
            if len(self.example_data) == self.max_num_shots:
                self.random_indice = list(range(len(self.example_data)))
            else:
                self.random_indice = np.random.choice(len(self.example_data), self.max_num_shots, replace=False)

        self.formatted_evaluation_data = [self._format_instance(data) for data in self.evaluation_data]

        # automatic instruction
        if self.ape is True:
            instrction = ape(
                self.formatted_example_data, self.formatted_evaluation_data, self.model.get_ppl, self.model.api_key
            )
            self.instruction = instrction

        construct_fn = getattr(
            self, "_construct_instances_" + self.model_evaluation_method.split("_")[-1],
            self._construct_instances_generation
        )
        evaluation_instances, option_nums = construct_fn()

        def _print(info, instance, idx):
            if isinstance(instance, str):
                instance = [instance]
            instance = [i for i in instance if isinstance(i, str)]
            if len(instance) <= 2:
                labels = ["source", "target"]
            else:
                labels = [f"source_{i}" for i in range(len(instance) - 1)] + ["target"]
            for i, seg in zip(range(len(instance)), labels):
                info(f"Formatted evaluation instance {idx} ({seg})\n" + pformat(instance[i], width=100))

        _print(logger.info, evaluation_instances[0], 0)
        if len(evaluation_instances) > 1:
            _print(logger.debug, evaluation_instances[1], 1)

        # for self-consistency
        evaluation_instances = evaluation_instances * self.sample_num
        option_nums = option_nums * self.sample_num
        if isinstance(evaluation_instances[0], str):
            self.total_prefix_num = 1
        else:
            self.total_prefix_num = len([1 for i in evaluation_instances[0] if isinstance(i, str)])
        if self.total_prefix_num <= 1 and self.prefix_caching:
            logger.warning(
                f"Setting prefix_caching to False, since the total prefix number is {self.total_prefix_num}."
            )
            self.prefix_caching = False
        return evaluation_instances, option_nums

    def _construct_instances_ppl(self) -> Tuple[List[Tuple[str, ...]], List[int]]:
        evaluation_instances = []
        option_nums = []
        for formatted_instance in self.formatted_evaluation_data:
            instance_with_examples = self.construct_instruction_and_examples(formatted_instance)
            if "options" in formatted_instance:
                if isinstance(instance_with_examples, list):
                    options = [(*instance_with_examples, option) for option in formatted_instance["options"]]
                    prefix_length = len(instance_with_examples)
                else:
                    options = [(instance_with_examples, option) for option in formatted_instance["options"]]
                    prefix_length = 1

                option_nums.append(len(options))

                if self.use_normalization:
                    options = self._apply_normalization(options, prefix_length - 1)

                evaluation_instances.extend(options)

            else:
                # multiple contexts instead of options, cases like winogrande
                contexts = [(context, formatted_instance["target"]) for context in instance_with_examples]
                evaluation_instances.extend(contexts)
                option_nums.append(len(contexts))
        return evaluation_instances, option_nums

    def _construct_instances_prob(self) -> Tuple[_InputsWithOptionNum, List[int]]:
        evaluation_instances = []
        option_nums = []
        for formatted_instance in self.formatted_evaluation_data:
            instance_with_examples = self.construct_instruction_and_examples(formatted_instance)
            option_num = len(formatted_instance["options"])
            option_nums.append(option_num)
            if isinstance(instance_with_examples, str):
                evaluation_instances.append((instance_with_examples, option_num))
            else:
                evaluation_instances.append(tuple(instance_with_examples) + (option_num,))
        return evaluation_instances, option_nums

    def _construct_instances_generation(self) -> Tuple[List[Tuple[str, ...]], List[Literal[1]]]:
        evaluation_instances = []
        option_nums = []
        for formatted_instance in self.formatted_evaluation_data:
            evaluation_instances.append(self.construct_instruction_and_examples(formatted_instance))
            if "options" in formatted_instance:
                option_num = len(formatted_instance["options"])
            else:
                option_num = 1
            option_nums.append(option_num)
        return evaluation_instances, option_nums

    def _format_instance(self, instance, loose: bool = False, format_example: bool = False):
        """Format the dataset instance into task source text, target text, and options (for ranking).

        Args:
            `instance (Dict)`: an instance dict of multiple key-value pairs.
            `loose (bool, optional)`: Whether to add extra newline characters. Defaults to False.
            `format_example (bool, optional):` Whether to format the example. This will only effect datasets like winogrande by returning the correct source only. Defaults to False.
        """
        # it is not recommended to modify instance, in case of multiple calls
        formatted_instance = self.format_instance(instance)
        loose = "\n" if loose else ""  # type: ignore

        if "target_idx" in formatted_instance:
            if self.ranking_with_options:
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
            elif self.model_evaluation_method == "generation":
                formatted_instance["target"] = chr(65 + target_idx)

        if "source_postfix" in formatted_instance:
            formatted_instance["source"] += formatted_instance.pop("source_postfix")

        # remove redundant spaces
        if isinstance(formatted_instance.get("target", None), str):
            formatted_instance["target"] = " " + formatted_instance["target"].lstrip()

        if format_example and "source_idx" in formatted_instance:
            formatted_instance["source"] = formatted_instance["source"][formatted_instance.pop("source_idx")]

        return formatted_instance

    def construct_instruction_and_examples(
        self,
        instance: Dict[str, typing.Any],
        split_prefix: Optional[bool] = None,
    ) -> Union[str, List[str]]:
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (dict): the pre-formatted source.

        Returns:
            Union[str, List[str]]: The final formatted instance. Return a list of formatted instances if the source is a list (in cases like winogrande).
        """
        if split_prefix is None:
            # vllm also supports prefix_caching but does not need to split the prefix
            split_prefix = self.prefix_caching and self.model.is_huggingface_model()

        if self.examples == "" or self.kate or self.globale:
            self.examples = self.construct_examples(instance)

        if self.model_type not in {"base", "instruction", "chat"}:
            raise ValueError(
                f"Invalid model type: {self.model_type}. Please use `--model_type` to specify the"
                " model type, which can be chosen from `base` and `instruction`."
            )

        _instruction = self.instruction + "\n\n" if len(self.instruction) > 0 else ""

        if isinstance(instance["source"], list):
            # return a list of formatted instances if the source is a list (in cases like winogrande)
            sources = [self.examples + self.instance_format.format(source=s, target="") for s in instance["source"]]
            if self.model_type == "instruction":
                sources = [_instruction + s for s in sources]

            return sources
        else:
            source = self.instance_format.format(source=instance["source"], target="")
            if self.model_type == "instruction":
                results = [_instruction, self.examples, source]
            else:
                results = [self.examples, source]
            if split_prefix:  # to support prefix_caching
                results = [p for p in results if len(p) > 0]
                return results
            else:
                return "".join(results)

    def construct_examples(self, instance: Optional[dict] = None) -> str:
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Dict): a pre-formatted evaluation instance.

        Returns:
            str: The constructed demonstration text.
        """
        if self.max_num_shots == 0:
            self.real_num_shots = 0
            self.real_example_tokens = 0
            return ""
        elif len(self.example_data) == 0:
            raise ValueError(
                f"Receive num_shots={self.max_num_shots}, but cannot construct examples for dataset {self.name} without example data."
            )

        if self.kate is True:
            assert instance is not None
            if isinstance(instance["source"], list):
                instance_source = instance["source"][instance["source_idx"]]
            else:
                instance_source = instance["source"]

            # select demonstrations based on knn algorithm
            # TODO: Bugs in kate, order, filter
            indice = knn_construct_examples(instance_source, self.formatted_example_data, self.max_num_shots)
        else:
            indice = self.random_indice

        if self.globale is True:
            # rank demonstrations based on global entropy
            labels = list(range(len(self.formatted_example_data[0]["options"])))
            indice = global_entropy_ordering_strategy(indice, labels, self.formatted_example_data, self.model.get_ppl)

        # construct few-shot examples
        if hasattr(self, "formatted_example_data"):
            examples = [self.formatted_example_data[i] for i in indice]
        else:
            examples = [self._format_instance(self.example_data[i], format_example=True) for i in indice]
        example_texts = [self.instance_format.format_map(example) + "\n\n" for example in examples]
        example_text, self.real_example_tokens, self.real_num_shots = self.truncate_by_word(
            words=example_texts,
            max_tokens=self.max_example_tokens,
            side="right",
        )
        if self.real_num_shots < self.max_num_shots:
            logger.warning(
                f"Only {self.real_num_shots} examples ({self.real_example_tokens} tokens) are constructed because of `--max_example_tokens {self.max_example_tokens}`, but the few-shot number is set to {self.max_num_shots}."
            )
        return example_text

    def _init_model(self) -> Optional[Dict[str, typing.Any]]:
        """(Re-)initialize the model before iterating through the dataset. This is useful when evaluating on a mixture of `GenerationDataset` and `MultipleChoiceDataset`. Call this function manuanlly before iterating the dataset, or use `DatasetCollectionBatchSampler` to manage the context switching automatically."""
        if getattr(self, "is_iter_initialized", False):
            return
        self.is_iter_initialized = True

        if self.model_evaluation_method == "get_prob":
            self._extra_model_args["constant_option_num"] = all(n == self.option_nums[0] for n in self.option_nums)
        elif self.model_evaluation_method == "generation" and self.evaluation_type == "ranking":
            self._extra_model_args = {}
            self._extra_model_args["max_tokens"] = 1
            self._extra_model_args["temperature"] = 0.0
            self._extra_model_args["stop"] = ["\n"]

        self.model._reload_tokenizer()
        if self.model_evaluation_method == "get_ppl":
            return self.model.set_ppl_args(**self._extra_model_args)
        elif self.model_evaluation_method == "generation":
            return self.model.set_generation_args(**self._extra_model_args)
        elif self.model_evaluation_method == "get_prob":
            return self.model.set_prob_args(**self._extra_model_args)

    def post_processing(self, predictions: Union[List[Tuple[str, float]], List[List[float]]]):
        r"""Post processing for the predictions.

        Args:
            predictions (List[Union[str, float]]): The predicted answers (generated texts or perplexity scores).

        Returns:
            List[Union[str, float]]: The post-processed predictions.
        """
        return predictions

    def calculate_metric(self, predictions) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]:
        r"""Calculate the metric score between `predictions` and `references`.

        Args:
            predictions (List[Union[int, str]]): The predicted answers.

        Returns:
            Dict[str, Dict[str, float]]: The metric results in the format `{"Dataset": {"Metric": Score}}` or `{"Dataset:Subset": {"Metric": Score}}`.
            Dict[str, List[float]]: The score lists. This is useful for logging result for each instance.
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
        # datasets like winogrander can be categorized by gender
        if self.category_column is not None:
            subject_results = pd.DataFrame({
                "predictions":
                predictions,
                "references":
                self.references,
                "subject":
                map(lambda i: f"{self.name}[{i[self.category_column]}]", self.evaluation_data),
            }).groupby("subject").apply(lambda df: _calculate_metric(df["predictions"], df["references"])).to_dict()

        metric_results = OrderedDict(**subject_results)
        metric_results[self.dataset_name] = overall_results
        return metric_results, score_lists

    def last_score_lists(self) -> Dict[str, List[float]]:
        results = {}
        for metric in self.metrics:
            results.update(metric.last_score_lists())
        return results

    def len(self, sample_num: bool = True, option_num: bool = True, normalization: bool = True) -> int:
        """Provides a unified interface to retrieve the length of dataset`.

        - `len(dataset.evaluation_data)` or `len(dataset.evaluation_data)`: the length of raw evaluation data
        - `len(dataset)` or `len(dataset.evaluation_instances)`: the length of `__iter__`. Equal to length of raw data multiplied by `self.sample_num`, option_num (if `model_evaluation_method` is "get_ppl") and 2 (if `use_normalization` is True)
        """
        # if `model_evaluation_method` is not "get_ppl", two branches of `option_num` should be equivalent
        if option_num:
            length = len(self.evaluation_instances)
            if not sample_num and self.sample_num > 1:
                length = length // self.sample_num
            if not normalization and self.use_normalization:
                length = length // 2
        else:
            length = len(self.evaluation_data)
            if sample_num and self.sample_num > 1:
                length *= self.sample_num
            if normalization and self.use_normalization:
                length *= 2
        return length

    def log_final_results(
        self,
        raw_predictions: List[str],
        processed_predictions: List[Union[str, float]],
        score_lists: Dict[str, List[float]],
    ) -> Optional[pd.Series]:
        return log_final_results(
            raw_predictions, processed_predictions, score_lists, self.name == "winogrande",
            self.model_evaluation_method, self.use_normalization, self.option_nums, self.evaluation_data,
            self.evaluation_instances, self.sample_num, self.references
        )

    def __repr__(self):
        reprs = [f"{p}={getattr(self, p)!r}" for p in self._repr]
        reprs.append(f"len={self.len()}")
        reprs.append(f"num_instances={self.len(sample_num=False, option_num=False, normalization=False)}")
        return "Dataset(" + ", ".join(reprs) + ")"


class DatasetCollection(torch.utils.data.Dataset):

    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__()
        self.dataset_names = list(datasets.keys())
        self._datasets = list(datasets.values())
        self._datasets_mapping = datasets
        self._cur_idx = 0
        self.args = self._datasets[0].args
        self._lines_iter = chain.from_iterable(
            zip(range(d.len()), d.evaluation_instances, repeat_iter(d.references, d.sample_num)) for d in self._datasets
        )

        self.categorized_subsets = {}
        for d in self._datasets:
            if d.categorized_subsets:
                self.categorized_subsets[d.name] = d.categorized_subsets

    @property
    def name(self) -> str:
        return self._datasets[self._cur_idx].name

    @property
    def option_nums(self) -> List[int]:
        """If `model_evaluation_method` is "get_ppl", this returns the total number of options across all evaluation examples. Otherwise, this returns an empty list."""
        return sum([d.option_nums for d in self._datasets], [])

    @property
    def strides(self) -> List[int]:
        """If `model_evaluation_method` is "get_ppl", this returns the total number of options across all evaluation examples. Otherwise, this returns an empty list."""
        option_nums = []
        for d in self._datasets:
            if d.use_normalization:
                o = [i * 2 for i in d.option_nums]
            else:
                o = d.option_nums
            if d.model_evaluation_method != "get_ppl":
                option_nums.extend([1] * len(o))
            else:
                option_nums.extend(o)
        return option_nums

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
    ) -> Iterator[Union[list, dict, None]]:
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

    def log_final_results(
        self,
        raw_predictions: List[str],
        processed_predictions: List[Union[str, float]],
        score_lists: List[Dict[str, List[float]]],
    ):
        lines = []
        raw = self._split_by_subset(raw_predictions)
        processed = self._split_by_subset(processed_predictions, option_num=False, normalization=False)

        for d, r, p, s in zip(self._datasets, raw, processed, score_lists):

            def set_subset(l: dict):
                l["subset"] = d.subset_name

            series = d.log_final_results(r, p, s)  # type: ignore
            if series is None:
                return
            series.apply(set_subset)
            lines.append(series)

        file = self.args.evaluation_results_path
        try:
            pd.concat(lines).to_json(file, orient="records", indent=4, force_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to log predictions: {e}")

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

    def calculate_metric(self, predictions) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, List[float]]]]:
        results = OrderedDict()
        score_lists = []
        splitted = self._split_by_subset(predictions, option_num=False, normalization=False, sample_num=False)
        grouped_dataset_names = defaultdict(list)  # group by dataset
        for n, d, p in zip(self.dataset_names, self._datasets, splitted):
            subset_results, score_list = d.calculate_metric(p)
            results.update(subset_results)
            score_lists.append(score_list)
            grouped_dataset_names[d.name].append(n)

        # calculate the mean of each category
        for name, dataset_names in grouped_dataset_names.items():
            if self.categorized_subsets.get(name, None):
                for cat, cat_subsets in self.categorized_subsets[name].items():
                    c = set(f"{name}:{s}" for s in cat_subsets)
                    if len(c.intersection(set(dataset_names))) != len(c):
                        # skip if not all subsets of a category are available
                        continue
                    fstr = f"{name}[{cat.title().replace('_', ' ')} Macro Average]"
                    results[fstr] = avg_metrics([results[n] for n in c])

            if name == "gaokao":
                r, f = zip(*[(results[name + ":" + n], f) for n, f in GAOKAO_CHINESE_TASKS_SCORE.items()])
                results[name + "[Chinese Weighted Average]"] = avg_metrics(r, f, average_method="weighted")
                r, f = zip(*[(results[name + ":" + n], f) for n, f in GAOKAO_ENGLISH_TASKS_SCORE.items()])
                results[name + "[English Weighted Average]"] = avg_metrics(r, f, average_method="weighted")
                r, f = zip(*[(results[name + ":" + n], f) for n, f in GAOKAO_TASKS_SCORE.items()])
                results[name + "[Weighted Average]"] = avg_metrics(r, f, average_method="weighted")

            results[name + "[Marco Average]"] = avg_metrics([r for k, r in results.items() if k.startswith(name + ":")])

        return results, score_lists

    def get_batch_sampler(self, reload_tokenizer: bool = False):
        if reload_tokenizer:
            self._datasets[0].model._remove_tokenizer()
        return DatasetCollectionBatchSampler(
            self, self.args.batch_size, self._datasets[0].model.model_backend == "vllm", self.args.auto_batch_size
        )

    def step(
        self,
        writer: PredictionWriter,
        tqdm: Union[dynamic_stride_tqdm, typing.Any],
        batch_raw_predictions: List[str],
    ):
        batch_size = len(batch_raw_predictions)
        if isinstance(tqdm, dynamic_stride_tqdm):
            tqdm.step(batch_size)
            if batch_size > 0:
                tqdm.set_description(self.dataset_names[self._cur_idx])
        if batch_size > 0:
            writer.log_batch_results(batch_raw_predictions, self._lines_iter)

    def __repr__(self):
        reprs = []
        reprs.append(f"dataset_names={self.dataset_names}")
        reprs.append(f"len={self.len()}")
        reprs.append(f"num_instances={self.len(sample_num=False, option_num=False, normalization=False)}")
        return "DatasetCollection(" + ", ".join(reprs) + ")"
