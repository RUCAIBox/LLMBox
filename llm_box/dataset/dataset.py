import os
from argparse import Namespace
from logging import getLogger
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Set, Union, Dict

import numpy as np
import torch
import datasets as d

from ..metric import MetricModule
from ..utils import import_main_class, NotImplementedField
from .raw_dataset_loader import load_raw_dataset, raw_dataset_config
from .utils import DataLoaderX

logger = getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    r"""The base class object for all datasets.

    Args:
        args (Namespace): The global configurations.

    Attributes:
        name (str): The name of this dataset.
        tokenizer (Union[transformers.PreTrainedTokenizer, tiktoken.Encoding]): The tokenizer of corresponding model.
        evaluation_data (List[dict]): The list of data for evaluation.
        evaluation_instances (List[Union[str, Tuple(str, str)]]): The list of formatted evaluation instances.
        evaluation_type (str): The method for evaluation, which can be set to either 'ranking' or 'generation'.
        metric (str): The metric for evaluating the predictions and references.
        instruction (str, *optional*): The instruction for this task.
        option_nums (List[int], *optional*): The list of the number of options for each instance (mainly used for multi-choice tasks).
        example_data (List[dict], *optional*): The list of demonstration data.
        num_shots (int, *optional*): The number of demonstration instances.
        max_example_tokens (int, *optional*): The number of maximum tokens in the demonstration.
    """

    _name: str = NotImplementedField
    _subset_name: Optional[Union[str, List[str]]] = NotImplementedField

    metrics: Set[str] = NotImplementedField
    evaluation_type: Literal['ranking', 'generation'] = NotImplementedField

    instruction: str = NotImplementedField

    example_set: Optional[str] = NotImplementedField
    evaluation_set: str = NotImplementedField

    def __init__(
        self,
        args: Namespace,
        subset_name: Optional[str] = None,
        raw_dataset: Optional[Union[d.DatasetDict, d.Dataset, d.IterableDataset, d.IterableDatasetDict]] = None,
    ):
        super().__init__()
        self.args = args

        if self._subset_name is None:
            self._subset_name = subset_name

        logger.debug(f"Raw dataset loaded: {raw_dataset}")
        if not isinstance(self.evaluation_set, str) or args.evaluation_set is not None:
            self.evaluation_set = args.evaluation_set
        if not isinstance(self.example_set, str) or args.example_set is not None:
            self.example_set = args.example_set
        if self.evaluation_set is None:
            raise ValueError("`evaluation_set` must be specified either in Dataset or in command line arguments.")
        msg = f"Using `{self.evaluation_set}` as evaluation_set"

        self.use_example = args.num_shots > 0 and self.example_set is not None and self.example_set in raw_dataset
        if self.use_example:
            msg += f" and `{self.example_set}` as example_set"
        logger.debug(msg)

        if isinstance(raw_dataset, (d.Dataset, d.IterableDataset)):
            self.evaluation_data = list(raw_dataset)
        elif isinstance(raw_dataset, (d.DatasetDict, d.IterableDatasetDict)):
            self.evaluation_data = list(raw_dataset[self.evaluation_set])
            if self.use_example:
                self.example_data = list(raw_dataset[self.example_set])
        else:
            raise ValueError(
                f"Cannot load Dataset from {raw_dataset}. If you are loading dataset with multiple subsets, try to load each subset seperately."
            )

        self.tokenizer = args.tokenizer

        self.num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.example_separator_string = args.example_separator_string
        self.examples = self.construct_examples()

        self.construct_instances()

        self.metric_modules = MetricModule(self.metrics)

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
            Dict:
                source: str
                target: str
                options (*optional*): List[str]
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `format_instance` function.")

    def format_instruction_and_examples(self, source, target=""):
        r"""Format one instance with the instruction and demonstration.

        Args:
            source (str): the pre-formatted source text.
            target (str, *optional*): the pre-formatted target text (default to "").

        Returns:
            Union[str, Tuple(str, str)]: The final formatted instance.
        """
        # TODO: instruction template
        # TODO: ICL
        if self.model.type == 'base':
            source = self.examples + source
        elif self.model.type == 'instruction':
            source = self.instruction + "\n\n" + self.examples + source

        if target:
            return source, target
        else:
            return source

    def construct_examples(self, instance=None):
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Dict): a pre-formatted evaluation instance.

        Returns:
            str: The constructed demonstration text.
        """
        # No example
        if not self.use_example:
            return ""

        # selection algorithm
        # TODO: ICL
        indice = np.random.choice(len(self.example_data), self.args.num_shots)

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

    def calculate_metrics(self) -> Dict[str, float]:
        r"""Calculate the metric score betwwen `predictions` and `references`.

        Args:
            predictions (List[str]): The predicted answers.

        Returns:
            Dict[str, float]: The metric results.
        """
        return self.metric_modules.calculate_metric()

    def get_dataloader(self, **kwargs):
        default_kwargs = dict(
            batch_size=self.args.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
            pin_memory=True,
        )
        default_kwargs.update(kwargs)
        return DataLoaderX(self, **default_kwargs)

    @property
    def name(self):
        return self._name + (f":{self._subset_name}" if self._subset_name else "")

    def __repr__(self):
        content = f"{self.name}[{self.evaluation_set}"
        if self.use_example:
            content += f", {self.example_set}"
        content += "]"
        return f"{self.__class__.__name__}({content})"


def load_dataset(
    args: Namespace,
    dataset_name_or_path: Union[str, Path],
    subset_names: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    methods: Union[str, List[str]] = "load_dataset",
    **kwargs
) -> Union[Iterator[Dataset], List[Dataset]]:
    r"""Load corresponding dataset class.

    Args:
        dataset (str): The name of dataset.

    Returns:
        Dataset: Our class for dataset.
    """
    # find the relative path from `main`
    dataset_name_or_path = str(os.path.normpath(dataset_name_or_path))

    if subset_names is None:
        # load from config, e.g. `copa`
        loaded_from_config = False
        for dataset_collection, subset_names in raw_dataset_config.items():
            if dataset_name_or_path in subset_names:
                loaded_from_config = True
                dataset_name = dataset_name_or_path
                dataset_path = dataset_collection
                subset_names = [dataset_name_or_path]
                break

        # load all subsets from path, e.g. `mmlu`, `super_glue`
        if not loaded_from_config:
            dataset_name = None
            dataset_path = dataset_name_or_path
            subset_names = None

    else:
        # load specific subsets from path,
        # e.g. `super_glue:copa` and `mmlu:abstract_algebra`
        dataset_name = None
        dataset_path = dataset_name_or_path
        subset_names = [subset_names] if isinstance(subset_names, str) else subset_names

    # load all subsets from dataset
    logger.debug(f"Loading raw dataset from {dataset_path} - {subset_names} with arguments {kwargs}")
    filter = lambda obj: isinstance(obj._name, str)
    subset_cls = True
    for n in [dataset_name, dataset_path.split("/")[-1]]:
        print(n, __name__)
        if n is not None:
            try:
                dataset_cls = import_main_class('..' + n, Dataset, __name__, filter)
                logger.debug(f"Dataset class `{dataset_cls.__name__}` loaded.")
                subset_cls = False
            except Exception:
                continue
            break

    # load LLMDataset for each subset
    results = []
    raw_datasets = load_raw_dataset(
        dataset_path=dataset_path, subset_names=subset_names, split=split, methods=methods, **kwargs
    )
    for subset_name, raw_dataset in raw_datasets.items():
        if subset_cls:
            dataset_cls = import_main_class('.' + subset_name, Dataset, __name__, filter)
        dataset = dataset_cls(args=args, subset_name=subset_name, raw_dataset=raw_dataset)
        results.append(dataset)

    return results
