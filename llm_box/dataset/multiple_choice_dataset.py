import numpy as np

from .dataset import Dataset


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score.
    """

    evaluation_type = "ranking"
    metric = "accuracy"

    def __init__(self, args, model):
        super().__init__(args, model)

    def calculate_metric(self, results):
        labels = []
        st = 0
        results = np.array([result / length for result, length in results])
        for num in self.option_nums:
            labels.append(results[st:st + num].argmin())
            st += num
        results = labels
        assert len(results) == len(self.references)

        score_list = np.asarray(results) == np.asarray(self.references)
        return {'Accuracy': np.mean(score_list)}
