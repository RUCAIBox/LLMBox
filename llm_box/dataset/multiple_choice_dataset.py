import numpy as np

from .dataset import Dataset


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score.
    """

    evaluation_type = "ranking"
    metric = "accuracy"

    def post_processing(self, results):
        labels = []
        st = 0
        results = np.array([result / length for result, length in results])
        for num in self.option_nums:
            labels.append(results[st:st + num].argmin())
            st += num
        results = labels


        return results

    def calculate_metric(self, results):
        score_list = np.asarray(results) == np.asarray(self.references)
        return {'Accuracy': np.mean(score_list)}
