import os
from logging import getLogger
from os.path import abspath
from typing import Dict, List, Optional, Tuple, Union, Literal
from pprint import pformat

import datasets as d
import numpy as np
import torch

from utils import *
from instance_utils import ape,global_entropy_ordering_strategy,knn_construct_examples
from ..utils import DatasetArguments, NotImplementedField
from ..model.model import Model

logger = getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    r"""The base class representing a dataset for a specific task.

    Class Attributes:
        - `name (str)`: The name of this dataset.
        instruction (str): Dataset-specific instruction for this task.
        - `metric (str)`: The metric used for evaluation.
        - `evaluation_type (Literal['ranking', 'generation'])`: The type of evaluation for the dataset.
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

    name: str = NotImplementedField
    r"""The name of this dataset. Should be identical to the file name."""

    instruction: str = NotImplementedField
    r"""Dataset-specific instruction for the task."""

    metrics: List = NotImplementedField
    r"""The metric functions used for evaluation."""

    evaluation_type: Literal['ranking', 'generation'] = NotImplementedField
    r"""The type of evaluation for the dataset."""

    evaluation_set: str = NotImplementedField
    r"""The evaluation split of dataset. Evaluation data will be automatically loaded."""

    example_set: Optional[str] = NotImplementedField
    r"""The example split of dataset. Example data will be automatically loaded if this is not None."""

    load_args: Union[Tuple[str], Tuple[str, str]] = NotImplementedField
    r"""Arguments for loading the dataset with huggingface `load_dataset`.

    Supported formats:
        - `(dataset_name,)`: If the dataset supports specifying subset name from command line, or only has one subset. E.g., `('race',)` and `('hendrycks/competition_math',)` respectively.
        - `(dataset_name, subset_name)`: If the dataset itself is a subset of a dataset collection. E.g., `('super_glue', 'copa')`.
    """

    def __init__(self, args: DatasetArguments, model: Model, subset_name: Optional[str] = None):
        r"""This should be called by the subclass.

        Args:
            - `args (DatasetArguments)`: The arguments for the dataset.
            - `model (Model)`: Our class for model.
            - `subset_name (Optional[str])`: The subset name of the dataset. Used when loading raw dataset from huggingface.
        """
        super().__init__()
        self.args = args

        self._load_raw_dataset(
            dataset_path=args.dataset_path,
            subset_name=subset_name,
            evaluation_set=args.evaluation_set or self.evaluation_set,
            example_set=args.example_set or self.example_set,
        )

        self.model = model
        self.tokenizer = model.tokenizer

        self.num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.examples = self.construct_examples()
        self.construct_instances()

    def _load_raw_dataset(
        self,
        dataset_path: Optional[str],
        subset_name: Optional[str],
        evaluation_set: str,
        example_set: Optional[str],
    ):
        r"""Load the raw dataset from huggingface or local path into `self.evaluation_data` and `self.example_data`. If `dataset_path` is not None, the dataset will be loaded from the given local path."""

        msg = f"Loading raw dataset `{self.name}:{subset_name}`"
        if dataset_path is not None:
            loadders = []
            dataset_path = abspath(dataset_path)
            msg += f" from local path `{dataset_path}`"
            if os.path.exists(dataset_path + "/dataset_infos.json"):
                loadders.append(lambda s: d.load_dataset(dataset_path, self.name, split=s))
                loadders.append(lambda s: d.load_dataset(dataset_path, subset_name, split=s))
            elif os.path.exists(dataset_path + "/dataset_dict.json"):
                loadders.append(lambda s: d.load_from_disk(dataset_path)[s])
            elif isinstance(subset_name, str) and os.path.exists(f"{dataset_path}/{subset_name}/dataset_dict.json"):
                loadders.append(lambda s: d.load_from_disk(dataset_path + "/" + subset_name)[s])
        else:
            if len(self.load_args) == 1 and isinstance(subset_name, str):
                load_args = self.load_args + (subset_name,)
            elif isinstance(subset_name, str):
                raise ValueError(
                    f"Failed to specify {subset_name} subset since dataset `{self.name}` already has defined one to load ({', '.join(self.load_args)}). Please use `{self.name}`."
                )
            else:
                load_args = self.load_args
            msg += f" from huggingface ({', '.join(load_args)})"
            loadders = [lambda s: d.load_dataset(*load_args, split=s)]

        logger.debug(
            msg + f" with evaluation set `{evaluation_set}`" +
            (f" and example set `{example_set}`" if example_set else "")
        )

        for fn in loadders:
            try:
                self.evaluation_data = list(fn(evaluation_set))
                self.example_data = list(fn(example_set)) if example_set else []
                logger.info(
                    f"Evaluation data with {len(self.evaluation_data)} instances:\n{pformat(self.evaluation_data[0])}"
                )
                return
            except KeyError as e:
                raise ValueError(f'Unknown split "{e}" of dataset "{self.name}"')
            except ValueError as e:
                # catch unknown split error raise from load_dataset
                if "Unknown split" in str(e):
                    raise e
            except Exception as e:
                # continue to try other loaders
                logger.info(f"{e.__class__.__name__} when loading dataset: {e}")

        raise ValueError("Failed to load" + msg[7:])

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

    def format_instance(self, instance):
        r"""Format the dataset instance into task source text, target text, and options (for ranking).

        Args:
            instance (Dict): an instance dict of multiple key-value pairs.

        Returns:
            Dict: A dictionary containing the formatted instance.
                source (str): The source text.
                target (str): The target text.
                options (List[str], optional): The options for ranking.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `format_instance` function.")

    def format_instruction_and_examples(self, source, target=""):
        r"""Format one instance with the instruction and demonstration.

        Args:
            source (str): the pre-formatted source text.
            target (str, optional): the pre-formatted target text (default to "".

        Returns:
            Union[str, Tuple(str, str)]: The final formatted instance.
        """
        # TODO: instruction template

        # TODO: ICL

        if self.model.type == 'base':
            source = self.examples + source
        elif self.model.type == 'instruction':
            preformatted_example_dataset = formatted_copa_data(self.example_data)
            preformatted_evaluation_dataset = formatted_copa_data(self.evaluation_data)
            # construct instructions using ape
            instrctions,instructions_scores = ape(preformatted_example_dataset,preformatted_evaluation_dataset)
            self.instruction = instrctions[0]
            source = self.instruction + "\n\n" + self.examples + source

        if target:
            return source, target
        else:
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

        # selection algorithm
        # TODO: ICL
        # select demonstrations based on knn algorithm
        preformatted_dataset = formatted_copa_data(self.example_data)
        knn_indice = knn_construct_examples(instance["source"],preformatted_dataset,self.num_shots)

        # rank demonstrations based on global entropy
        labels = []
        for i in range(len(self.example_data)):
            labels.append(str(self.example_data[i]["label"]))
        labels = list(set(labels))
        indice = global_entropy_ordering_strategy(knn_indice,labels,preformatted_dataset)
        # indice = np.random.choice(len(self.example_data), self.args.num_shots)

        # TODO: tokenizer efficiency
        # construct few-shot examples
        example_text = ""
        example_token_nums = 0
        for index in indice:
            example = self.format_instance(self.example_data[index])
            cur_example_text = self.args.instance_format.format_map(example) + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + example_token_nums <= self.max_example_tokens:
                example_text += cur_example_text
                example_token_nums += cur_token_num

        logger.info("Few-shot examples:\n" + pformat(example_text))
        return example_text

    def construct_instances(self):
        r"""Construct and format all the instances of `evaluation_data`.

        Returns:
            List[str]: The list of final formatted instances.
        """
        self.evaluation_instances = []
        self.option_nums = []
        for instance in self.evaluation_data:
            formatted_instance = self.format_instance(instance)
            if self.evaluation_type == "ranking":
                options = [
                    self.format_instruction_and_examples(formatted_instance["source"], option)
                    for option in formatted_instance['options']
                ]
                self.evaluation_instances.extend(options)
                self.option_nums.append(len(options))
            elif self.evaluation_type == "generation":
                self.evaluation_instances.append(self.format_instruction_and_examples(formatted_instance["source"]))
        self.evaluation_instances = self.evaluation_instances * self.args.sample_num
        self.option_nums = self.option_nums * self.args.sample_num

    def post_processing(self, predictions):
        r"""Process the generated predictions. For ranking-based datasets, it chooses the option with lowest PPL.
        For generation-based datasets, it may remove blank characters.

        Args:
            predictions (List[Union[float, str]]): the calculated PPL scores or predicted answers.
        
        Returns:
            List[Union[float, str]]: the processed results.
        """

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
        return f"DatasetCollection({self._datasets})"
