import importlib


def load_dataset(args, model):
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Returns:
        Dataset: Our class for dataset.
    """
    args.dataset = args.dataset.split(":")
    dataset = importlib.import_module(f"dataset.{args.dataset[0]}")
    dataset = getattr(dataset, args.dataset[0].capitalize())(args, model)
    return dataset
