import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from .dataset import Dataset as LLMDataset
from ..utils import import_main_class

__all__ = ['DataLoaderX', 'load_dataset']


@property
def NotImplementedField(self):
    raise NotImplementedError(f"{self.__class__.__name__} has not implemented field.")


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_dataset(dataset: str, *args, **kwargs) -> LLMDataset:
    r"""Load corresponding dataset class.

    Args:
        dataset (str): The name of dataset.

    Returns:
        Dataset: Our class for dataset.
    """
    # find the relative path from `main`
    dataset_cls = import_main_class('dataset.' + dataset, LLMDataset)
    dataset = dataset_cls(*args, **kwargs)
    return dataset


