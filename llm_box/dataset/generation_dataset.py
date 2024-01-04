from .dataset import Dataset


class GenerationDataset(Dataset):
    r"""The dataset for Generation problems. It solves problems in natural language and is evaluated using `accuracy` score.
    """

    evaluation_type = "generation"
    metric = "accuracy"

