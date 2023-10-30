from argparse import Namespace
from typing import List, Union, Optional, Dict
from pathlib import Path

import datasets
from datasets import DatasetDict

from .multiple_choice_dataset import MultipleChoiceDataset
from .raw_dataset_loader import get_dataset_subset_names, register_raw_dataset_loader


@register_raw_dataset_loader
def load_origin_mmlu(
    dataset_path: Union[str, Path] = None,
    subset_names: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    **kwargs
) -> Dict[str, DatasetDict]:
    """An example loader for the original MMLU dataset."""

    # get subset to load (if not specified, load all)
    dataset_subset = get_dataset_subset_names(dataset_path, local=True)
    if subset_names is None:
        subset_names = dataset_subset
    elif isinstance(subset_names, str):
        subset_names = [subset_names]
    subset = set(dataset_subset) & set(subset_names)
    splits = [split] if split is not None else ['dev', 'test']
    files = {
        f"{s}.{split}": f"{dataset_path}/{split}/{s}_{split}.csv"
            for split in splits for s in subset
    }

    # load all files at once to accelerate I/O
    raw_dataset = datasets.load_dataset(
        "csv",
        data_files=files,
        header=None,
        **getattr(kwargs, "load_dataset_kwargs", dict())
    )
    processed_dataset = raw_dataset.map(lambda x: {
        'question': x['0'],
        'choices': [x['1'], x['2'], x['3'], x['4']],
        'labels': x['5'],
    }, remove_columns=['0', '1', '2', '3', '4', '5'])

    dataset = {
        s: DatasetDict(
            { split: processed_dataset[f"{s}.{split}"] for split in splits }
        ) for s in subset
    }
    return dataset


class MMLU(MultipleChoiceDataset):
    """The dataset of MMLU.

    This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. This covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability.

    Example:
    """
    # TODO: Add exmaple

    _name = "mmlu"
    _subset_name = None
    example_set = "dev"
    evaluation_set = "test"

    def format_instance(self, instance):

        return dict(ground_truth=("test", "a"), options=[])

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
