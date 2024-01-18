import json
from logging import getLogger
from pprint import pformat
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from ..model.model import Model
from ..utils import DatasetArguments
from .icl_strategies import ape, global_entropy_ordering_strategy, knn_construct_examples
from .utils import get_raw_dataset_loader

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
        - `load_args (Union[Tuple[str], Tuple[str, str]])`: Arguments for loading the dataset with huggingface `load_dataset`.

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

    metrics: List
    r"""The metric functions used for evaluation."""

    evaluation_type: Literal['ranking', 'generation', 'user-defined']
    r"""The type of evaluation for the dataset."""

    evaluation_set: str
    r"""The evaluation split of dataset. Evaluation data will be automatically loaded."""

    example_set: Optional[str]
    r"""The example split of dataset. Example data will be automatically loaded if this is not None."""

    load_args: Union[Tuple[str], Tuple[str, str]]
    r"""Arguments for loading the dataset with huggingface `load_dataset`.

    Supported formats:
        - `(dataset_name,)`: If the dataset supports specifying subset name from command line, or only has one subset. E.g., `('race',)` and `('hendrycks/competition_math',)` respectively.
        - `(dataset_name, subset_name)`: If the dataset itself is a subset of a dataset collection. E.g., `('super_glue', 'copa')`.
    """

    model_args: Dict[str, Any] = dict()
    """Arguments for the model generation or get_ppl. See `set_generation_args` or `set_ppl_args` for details."""

    _repr = [
        "name", "subset_name", "instruction", "metrics", "evaluation_type", "evaluation_set", "example_set",
        "load_args", "model_args"
    ]

    def __init__(self, args: DatasetArguments, model: Model, subset_name: Optional[str] = None):
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

        if self.args.sample_num > 1 and self.evaluation_type == 'ranking':
            self.args.sample_num = 1
            logger.warning(
                f"Self-consistency only supports evaluation using the generation mode, automatically set sample_num=1."
            )

        if self.args.sample_num > 1 and self.evaluation_type == 'generation' and (
            getattr(self.model.args, 'temperature') == 0 or
            (getattr(self.model.args, 'temperature') == None and self.model_args.get('temperature', 1) == 0)
        ):
            self.model.args['temperature'] = 1
            logger.warning(
                f"Self-consistency only supports generation with temperature>0, automatically set temperature=1."
            )

        self.load_raw_dataset(
            dataset_path=args.dataset_path,
            subset_name=subset_name,
            evaluation_set=args.evaluation_set or self.evaluation_set,
            example_set=args.example_set or self.example_set,
        )

        self.num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.examples = ""
        self.kate = args.kate
        self.globale = args.globale
        self.ape = args.ape
        if self.args.num_shots:
            if self.ape or self.kate or self.globale:
                self.formatted_example_data = [self.format_instance(data) for data in self.example_data]
            if len(self.example_data) < self.args.num_shots:
                logger.warning(
                    f"The example data only has {len(self.example_data)} instances, but the few-shot number is set to {self.args.num_shots}. Setting the few-shot number to {len(self.example_data)}."
                )
                self.args.num_shots = len(self.example_data)
            if len(self.example_data) == self.args.num_shots:
                self.random_indice = list(range(len(self.example_data)))
            else:
                self.random_indice = np.random.choice(len(self.example_data), self.args.num_shots, replace=False)
        self.formatted_evaluation_data = [self.format_instance(data) for data in self.evaluation_data]
        self.construct_instances()

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
            return_msg=True
        )  # type: ignore
        logger.info(
            msg + f" with evaluation set `{evaluation_set}`" +
            (f" and example set `{example_set}`" if example_set else "")
        )

        self.evaluation_data = list(load_fn(evaluation_set))
        if self.args.num_shots:
            self.example_data = list(load_fn(example_set)) if example_set else []

        logger.info(f"Evaluation data with {len(self.evaluation_data)} instances")
        logger.info(f"The example instance:\n{pformat(self.evaluation_data[0])}")

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
        for formatted_instance in self.formatted_evaluation_data:
            if self.evaluation_type == "ranking":
                instance_with_examples = self.format_instruction_and_examples(formatted_instance)
                options = [(instance_with_examples, option) for option in formatted_instance['options']]
                self.evaluation_instances.extend(options)
                self.option_nums.append(len(options))
            elif self.evaluation_type == "generation":
                self.evaluation_instances.append(self.format_instruction_and_examples(formatted_instance))
        if self.evaluation_type == "ranking":
            logger.info("Evaluation mode: calculate PPL of the optional text based on the source text")
            logger.info("Formatted example (source)\n" + self.evaluation_instances[0][0])
            logger.info("Formatted example (option)\n" + self.evaluation_instances[0][1])
        else:
            logger.info("Evaluation mode: generation based on the source text")
            logger.info("Formatted example (source)\n" + self.evaluation_instances[0])

        # for self-consistency
        self.evaluation_instances = self.evaluation_instances * self.args.sample_num
        self.option_nums = self.option_nums * self.args.sample_num

    def format_instance(self, instance):
        r"""Format the dataset instance into task source text, target text, and options (for ranking).

        Notes:
            The instance should not be mutated since the function might be called for multiple times when formatting examples.

        Args:
            instance (Dict): an instance dict of multiple key-value pairs.

        Returns:
            Dict: A dictionary containing the formatted instance.
                source (str): The source text.
                target (str): The target text.
                options (List[str], optional): The options for ranking.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `format_instance` function.")

    def format_instruction_and_examples(self, instance) -> str:
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (dict): the pre-formatted source.

        Returns:
            str: The final formatted instance.
        """
        if self.examples == '' or self.kate or self.globale:
            self.examples = self.construct_examples(instance)
        if self.model.type == 'base':
            source = self.examples + self.args.instance_format.format(source=instance["source"], target="")
        elif self.model.type == 'instruction':
            source = self.instruction + "\n\n" + self.examples + self.args.instance_format.format(
                source=instance["source"], target=""
            )

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
            # select demonstrations based on knn algorithm
            # TODO: Bugs in kate, order, filter
            indice = knn_construct_examples(instance["source"], self.formatted_example_data, self.num_shots)
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
            if hasattr(self, 'formatted_example_data'):
                example = self.formatted_example_data[index]
            else:
                example = self.format_instance(self.example_data[index])
            cur_example_text = self.args.instance_format.format_map(example) + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + example_token_nums <= self.max_example_tokens:
                example_text += cur_example_text
                example_token_nums += cur_token_num

        return example_text

    def calculate_metric(self, predictions) -> Dict[str, float]:
        r"""Calculate the metric score between `predictions` and `references`.

        Args:
            predictions (List[Union[int, str]]): The predicted answers.

        Returns:
            Dict[str, float]: The metric results.
        """
        results = {}
        for metric_func in self.metrics:
            results.update(metric_func(predictions, self.references))
        return results

    def post_processing(self, predictions: List[Union[str, float]]):
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
        file: Optional[str] = None,
    ):
        r"""Save the dataset inputs and corresponding model predictions to file.

        Args:
            raw_predictions (List[str]): The raw predictions of model.
            processed_predictions (Optional[List[Union[str, float]]]): The processed answers.
            file (Optional[str]): The file path to save the predictions. If None, will use `args.evaluation_results_path`.
        """
        file = file or self.args.evaluation_results_path
        if self.evaluation_type == "generation" and processed_predictions is not None:
            # log intermediate and post-processed results
            # TODO , add perplexity
            dataset_info = (self.evaluation_instances, processed_predictions)
            keys = ["index", "input", "processed_prediction", "raw_prediction", "reference"]
        elif self.evaluation_type == "generation" and processed_predictions is None:
            # log intermediate results only
            dataset_info = (self.evaluation_instances,)
            keys = ["index", "input", "raw_prediction", "reference"]
        else:  # ranking
            # TODO group by question
            indices = [(i, j) for i in range(len(self.option_nums)) for j in range(self.option_nums[i])]
            question_index, option_index = zip(*indices)
            source_text, target_text = zip(*self.evaluation_instances)

            dataset_info = [question_index, option_index, source_text, target_text]
            keys = ["index", "question_index", "option_index", "source", "target", "perplexity", "reference"]

        def repeat_iter(obj, n):
            for _ in range(n):
                yield from obj

        lines = zip(
            range(len(self)),
            *dataset_info,
            raw_predictions,
            repeat_iter(self.references, self.args.sample_num),
        )

        lines = [dict(zip(keys, line)) for line in lines]
        with open(file, "w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2, ensure_ascii=False)

    def __repr__(self):
        return "Dataset(" + ", ".join(f"{p}={pformat(getattr(self, p))}" for p in self._repr) + ")"


class DatasetCollection(torch.utils.data.Dataset):

    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__()
        self._subset_names = list(datasets.keys())
        self._datasets = list(datasets.values())
        self._cur_idx = 0

    @property
    def name(self) -> str:
        return self._datasets[0].name

    def __len__(self):
        return sum(len(d) for d in self._datasets)

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise IndexError(f"Index {idx} out of range")
        self._cur_idx = 0
        while idx >= len(self._datasets[self._cur_idx]):
            idx -= len(self._datasets[self._cur_idx])
            self._cur_idx += 1
        return self._datasets[self._cur_idx][idx]

    def __iter__(self):
        for self._cur_idx, d in enumerate(self._datasets):
            yield from d.__iter__()

    def __getattr__(self, attr):
        return getattr(self._datasets[self._cur_idx], attr)

    def calculate_metric(self, predictions) -> Dict[str, Dict[str, float]]:
        results = dict()
        cur_len = 0
        for s, d in zip(self._subset_names, self._datasets):
            results[d.name + ":" + s] = d.calculate_metric(predictions[cur_len:cur_len + len(d)])
            cur_len += len(d)
        return results

    def __repr__(self):
        return "DatasetCollection(" + ", ".join(str(d) for d in self._datasets) + ")"
