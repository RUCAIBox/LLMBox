import numpy as np
import sacrebleu
from .dataset import Dataset


class TranslationDataset(Dataset):
    r"""The dataset for translation tasks. It ranks given options and is evaluated using `sacrebleu.sentence_bleu` score.
    """

    evaluation_type = "ranking"
    metric = "sacrebleu.sentence_bleu"

    def __init__(self, args):
        super().__init__(args)

    def calculate_metric(self, predictions):
        scores = []
        for prediction, reference in zip(predictions, self.references):
            bleu = sacrebleu.sentence_bleu(prediction, [reference])
            scores.append(bleu.score)
        return {'Average SacreBleu sentences score': np.mean(scores)}
