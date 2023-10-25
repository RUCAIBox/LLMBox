import importlib


def load_dataset(args):
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.

    Returns:
        Dataset: Our class for dataset.
    """
    dataset = importlib.import_module(f"dataset.{args.dataset}")
    dataset = getattr(dataset, args.dataset.capitalize())(args)
    return dataset
