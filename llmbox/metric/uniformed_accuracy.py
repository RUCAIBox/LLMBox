from typing import Dict
import numpy as np
import tiktoken

from .metric import Metric


class Uniformed_Accuracy(Metric):
    r""" For those tasks only require to predict curtain number words, calculate the Accuracy score.
    
    Return
        "Uniformed_Accuracy": float
    """

    def __call__(self, predictions, references) -> Dict[str, float]:
        self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_name_for_model('davinci-002'))

        encoded_references = self.tokenizer.encode_batch(references)
        length = [len(_) for _ in encoded_references]
        encoded_predictions = self.tokenizer.encode_batch(predictions)
        encoded_predictions = [encoded_pred[:l] for encoded_pred, l in zip(encoded_predictions, length)]

        predictions, references = self.tokenizer.decode_batch(encoded_predictions
                                                              ), self.tokenizer.decode_batch(encoded_references)
        predictions = [pred.lower() for pred in predictions]

        score_list = np.asarray(predictions) == np.asarray(references)
        self._last_score_lists = {'Uniformed_Accuracy': score_list}
        return {'Uniformed_Accuracy': np.mean(score_list) * 100}
