from typing import Set

from .dataset import Dataset


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score.
    """

    evaluation_type = "ranking"
    metrics = {"accuracy"}
