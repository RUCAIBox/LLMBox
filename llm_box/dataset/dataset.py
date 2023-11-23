import os
from logging import getLogger
from os.path import normpath
from typing import Dict, Optional, Tuple, Union
from pprint import pformat

import datasets as d
import numpy as np
import torch

from ..utils import DatasetArguments
from ..model.model import Model

logger = getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    r"""The base class object for all datasets.

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
    name = ""
    metric = ""
    instruction = ""
    evaluation_type = ""

    evaluation_set: str = ""
    example_set: Optional[str] = None

    load_args: Union[Tuple[str], Tuple[str, str]] = None

    def __init__(self, args: DatasetArguments, model: Model):
        super().__init__()
        self.args = args

        self._load_raw_dataset(
            dataset_path=args.dataset_path,
            subset_name=args.subset_name,
            evaluation_set=args.evaluation_set or self.evaluation_set,
            example_set=args.example_set or self.example_set,
        )

        self.model = model
        self.tokenizer = model.tokenizer

        self.num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.examples = self.construct_examples()
        self.construct_instances()

    def _load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        """Load the raw dataset from huggingface or local path."""
        logger.debug(
            f"Loading raw dataset `{self.name}:{subset_name}`" + (f" from `{dataset_path}`" if dataset_path else "") +
            f" with evaluation set `{evaluation_set}`" + (f" and example set `{example_set}`" if example_set else "")
        )
        if len(self.load_args) == 1:
            load_args = self.load_args + (subset_name,)

        if dataset_path is not None:
            loadders = []
            dataset_path = "./" + normpath(dataset_path)
            if os.path.exists(dataset_path + "/dataset_infos.json"):
                loadders.append(lambda s: d.load_dataset(dataset_path, split=s))
            elif os.path.exists(dataset_path + "/dataset_dict.json"):
                loadders.append(lambda s: d.load_from_disk(dataset_path)[s])
            elif os.path.exists(dataset_path + "/" + subset_name + "/dataset_dict.json"):
                loadders.append(lambda s: d.load_from_disk(dataset_path + "/" + subset_name)[s])
        else:
            loadders = [lambda s: d.load_dataset(*load_args, split=s)]

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
                raise e
            except Exception as e:
                logger.info(f"{e.__class__.__name__} when loading dataset: {e}")

        msg = f"Cannot load dataset {self.name}"
        if dataset_path:
            raise ValueError(msg + f" from local path `{dataset_path}` or `{dataset_path}/{subset_name}`")
        else:
            raise ValueError(msg + f" from huggingface ({', '.join(load_args)})")

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

    def calculate_metric(self, predictions) -> Dict[str, float]:
        r"""Calculate the metric score betwwen `predictions` and `references`.

        Args:
            predictions (List[str]): The predicted answers.

        Returns:
            dict: The metric results.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `calcuate_metric` function.")


class DatasetCollection(torch.utils.data.Dataset):

    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__()
        self._subset_names = list(datasets.keys())
        self._datasets = list(datasets.values())
        self._cur_idx = 0

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
            yield from d

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
