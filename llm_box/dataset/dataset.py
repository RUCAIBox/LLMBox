import os
from logging import getLogger
from pathlib import Path
from typing import List, Literal, Optional, Union

import datasets as d
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..model.model import Model
from ..utils import DatasetArguments, NotImplementedField, import_subclass
from .raw_dataset_loader import RAW_DATASET_COLLECTIONS, load_raw_dataset

logger = getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    r"""The base class object for all datasets.

    A `Dataset` object contains a single dataset or a subset of a dataset collection (specified by `_name` and `_subset`). `load_dataset` is useful for loading a list of subsets at a time. The `raw_dataset` will be loaded into `self.evaluation_data` and optionally `self.example_data`. Data formatter such as `references` and `format_instance` should be correctly implemented in the subclass. `calculate_metric` should should as be implemented for evaluation.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

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
    """The name of this dataset. If the dataset is a subset of a collection, it should be set as the name of the collection (like `'super_glue'` or `'mmlu'`)."""

    _subset: str = NotImplementedField
    """The subset name of this dataset. It can be set as a default class variable (like `'copa'` in super_glue) or specified in `__init__` of subclass (like `'abstract_algebra'` in mmlu)."""

    metric: str = NotImplementedField
    evaluation_type: Literal['ranking', 'generation'] = NotImplementedField

    instruction: str = NotImplementedField
    """The instruction for this task. Set to manually `''` if not needed."""

    evaluation_set: str = NotImplementedField
    example_set: Optional[str] = NotImplementedField

    load_methods: List[str] = ['load_dataset', 'load_from_disk']

    def __init__(
        self,
        args: DatasetArguments,
        model: Model,
        subset_name: Optional[str] = None,
        raw_dataset: Optional[Union[d.DatasetDict, d.Dataset, d.IterableDataset, d.IterableDatasetDict]] = None,
    ):
        super().__init__()
        self.args = args
        self._check_fields_implementation()

        self.model = model
        self.model_type = model.type
        self.tokenizer = model.tokenizer

        if self._subset_name is None:
            self._subset_name = subset_name

        # Set evaluation split
        if not isinstance(self.evaluation_set, str) or args.evaluation_set is not None:
            self.evaluation_set = args.evaluation_set
        if self.evaluation_set is None:
            raise ValueError("`evaluation_set` must be specified either in Dataset or in command line arguments.")
        msg = f"Formatting {raw_dataset}: Using `{self.evaluation_set}` as evaluation_set"

        # Set example split
        if not isinstance(self.example_set, str) or args.example_set is not None:
            self.example_set = args.example_set
        if args.num_shots > 0 and self.example_set is not None and self.example_set in raw_dataset:
            self.use_example = True
            msg += f" and `{self.example_set}` as example_set"
        else:
            self.use_example = False

        # Load splits from raw dataset
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

        self.num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.examples = self.construct_examples()

        self.construct_instances()
        logger.debug(msg)

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
        if not self.use_example:
            return ""
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

    def calculate_metric(self, predictions):
        r"""Calculate the metric score betwwen `predictions` and `references`.

        Args:
            predictions (List[str]): The predicted answers.

        Returns:
            dict: The metric results.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `calcuate_metric` function.")

    def get_dataloader(self, **kwargs):
        default_kwargs = dict(
            batch_size=self.args.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
            pin_memory=True,
            padding_side="left",
        )
        default_kwargs.update(kwargs)
        return DataLoader(self, **default_kwargs)

    def _check_fields_implementation(self):
        r"""Check whether all required fields are implemented."""
        for field in ['metric', 'evaluation_type', 'instruction', 'evaluation_set', 'example_set']:
            try:
                getattr(self, field)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"{self.name} dataset must implement the `{field}` field. Set it as a class variable or specified during initalization."
                ) from e

    @property
    def name(self):
        return self._name + (f":{self._subset}" if self._subset else "")

    def __repr__(self):
        content = f"{self.name}[{self.evaluation_set}"
        if self.use_example:
            content += f", {self.example_set}"
        content += "]"
        return f"{self.__class__.__name__}({content})"


def load_dataset(
    args: DatasetArguments,
    model: Model,
    dataset_name_or_path: Union[str, Path],
    subset_names: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    methods: Optional[Union[str, List[str]]] = None,
) -> List[Dataset]:
    r"""Load corresponding dataset class. Raw dataset will be automatically loaded and formatted into `Dataset` class.

    Args:
        dataset (str): The name of dataset.

    Returns:
        List[Dataset]: A list of our class for dataset.
    """
    # find the relative path from `main`
    dataset_name_or_path = str(os.path.normpath(dataset_name_or_path))

    if subset_names is None:
        # load from config, e.g. `copa`
        loaded_from_config = False
        for dataset_collection, subset_names in RAW_DATASET_COLLECTIONS.items():
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
    logger.debug(f"Loading raw dataset from {dataset_path} - {subset_names}")
    subset_cls = True
    for n in [dataset_name, dataset_path.split("/")[-1]]:
        if n is not None:
            try:
                dataset_cls = import_subclass(
                    module_path='llm_box.dataset.' + n,
                    metaclass_type=Dataset,
                )
                logger.debug(f"Dataset class `{dataset_cls.__name__}` loaded.")
                subset_cls = False
            except Exception:
                continue
            break

    # fianlly load LLMDataset for each subset
    results = []
    if not subset_cls and methods is None:
        methods = subset_cls.load_methods
    elif methods is None:
        methods = ['load_datasets', 'load_from_disk']
    raw_datasets = load_raw_dataset(dataset_path=dataset_path, subset_names=subset_names, split=split, methods=methods)
    for subset_name, raw_dataset in raw_datasets.items():
        if subset_cls:
            dataset_cls = import_subclass(
                module_path='llm_box.dataset.' + subset_name,
                metaclass_type=Dataset,
            )
        dataset = dataset_cls(args=args, model=model, subset_name=subset_name, raw_dataset=raw_dataset)
        results.append(dataset)

    return results
