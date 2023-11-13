import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
from datasets import DatasetDict

from .multiple_choice_dataset import MultipleChoiceDataset
from .raw_dataset_loader import register_raw_dataset_loader


def get_dataset_subsets(dataset_path, by_split="test") -> List[str]:
    r"""Get the list of available subset names for a particular dataset such as MMLU."""
    files = os.listdir(os.path.join(dataset_path, by_split))
    filter = lambda f: f.endswith(f"_{by_split}.csv") and not f.startswith(".")
    dataset_subset = sorted({f.split("_test.csv")[0] for f in files if filter(f)})
    return dataset_subset


@register_raw_dataset_loader
def load_origin_mmlu(
    dataset_path: Union[str, Path],
    subsets: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    **kwargs
) -> Dict[str, DatasetDict]:
    """An example loader for the original MMLU dataset."""

    # get subset in folder to load (if not specified, load all)
    dataset_subset = get_dataset_subsets(dataset_path)
    if subsets is None:
        subsets = dataset_subset
    elif isinstance(subsets, str):
        subsets = [subsets]
    subset = set(dataset_subset) & set(subsets)
    splits = [split] if split is not None else ['dev', 'test']
    files = {f"{s}.{split}": f"{dataset_path}/{split}/{s}_{split}.csv" for split in splits for s in subset}

    # load all files at once to accelerate I/O
    raw_dataset = datasets.load_dataset(
        "csv", data_files=files, header=None, **getattr(kwargs, "load_dataset_kwargs", dict())
    )
    processed_dataset = raw_dataset.map(
        lambda x: {
            'question': x['0'],
            'choices': [x['1'], x['2'], x['3'], x['4']],
            'labels': x['5'],
        },
        remove_columns=['0', '1', '2', '3', '4', '5']
    )

    dataset = {s: DatasetDict({split: processed_dataset[f"{s}.{split}"] for split in splits}) for s in subset}
    return dataset


class Mmlu(MultipleChoiceDataset):
    """The dataset of MMLU.
    """

    _name = "mmlu"
    _subset = None

    instruction = ""

    example_set = "dev"
    evaluation_set = "test"

    load_methods = ['load_datasets', 'load_from_disk', 'load_origin_mmlu']

    def format_instance(self, instance):
        raise NotImplementedError(f"{self.name} dataset must implement the `format_instance` function.")

    @property
    def references(self):
        r"""Get the references for `evaluation_data`.

        Returns:
            List[str]: The list of ground-truth answers.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `references` property.")
