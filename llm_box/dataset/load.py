import importlib
import inspect
from logging import getLogger
from typing import Union

from datasets import get_dataset_config_names

from ..model.model import Model
from ..utils import DatasetArguments
from .dataset import Dataset, DatasetCollection

logger = getLogger(__name__)


def import_dataset_class(dataset_name: str) -> Dataset:
    module_path = __package__ + "." + dataset_name
    module = importlib.import_module(module_path)
    clsmembers = inspect.getmembers(module, inspect.isclass)

    for name, obj in clsmembers:
        if getattr(obj, "name", None) == dataset_name:
            logger.debug(f"Dataset class `{name}` imported from `{module_path}`.")
            return obj

    raise ValueError(
        f"Cannot find dataset class with name {dataset_name} in module {module_path}. "
        "Make sure the dataset class defines `name` attribute properly."
    )


def load_dataset(args: DatasetArguments, model: Model) -> Union[Dataset, DatasetCollection]:
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Returns:
        Dataset: Our class for dataset.
    """
    try:
        subset_names = get_dataset_config_names(args.dataset_name)
    except FileNotFoundError:
        subset_names = ["default"]
    except ConnectionError as e:
        logger.info(f"Receive {e.__class__.__name__}: {e}. Use `subset_name` instead.")
        subset_names = [args.subset_name]

    dataset_cls = import_dataset_class(args.dataset_name)
    if len(dataset_cls.load_args) == 1 and len(subset_names) > 1 and args.subset_name is None:
        logger.info(f"Loading subsets of dataset `{args.dataset_name}`: " + ", ".join(subset_names))
        datasets = {}
        for s in subset_names:
            args.subset_name = s
            datasets[s] = dataset_cls(args, model)
        args.subset_name = None
        return DatasetCollection(datasets)
    else:
        logger.info(f"Loading dataset `{args.dataset_name}`")
        return dataset_cls(args, model)
