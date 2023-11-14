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
    logger.info(f"Loading dataset `{args.dataset}`.")
    args.dataset = args.dataset.split(":")
    dataset = importlib.import_module(f".{args.dataset[0]}", package="llm_box.dataset")
    dataset = getattr(dataset, args.dataset[0].capitalize())(args, model)
    return dataset
