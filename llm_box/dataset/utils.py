import importlib
from logging import getLogger

logger = getLogger(__name__)


def load_dataset(args, model):
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Returns:
        Dataset: Our class for dataset.
    """
    logger.info(f"Loading dataset `{args.dataset_name}`.")
    dataset = importlib.import_module(f"dataset.{args.dataset_name}")
    dataset = getattr(dataset, args.dataset_name.capitalize())(args, model)
    return dataset
