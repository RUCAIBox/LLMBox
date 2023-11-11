import importlib


def load_dataset(args, model):
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Returns:
        Dataset: Our class for dataset.
    """
    dataset = importlib.import_module(f"dataset.{args.dataset}")
    dataset = getattr(dataset, args.dataset.capitalize())(args, model)
    return dataset
