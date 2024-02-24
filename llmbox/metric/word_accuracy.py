from typing import Dict
import numpy as np
import tiktoken

from .metric import Metric


class Word_Accuracy(Metric):
    r""" For those tasks only require to predict curtain number words, calculate the Accuracy score.
    
    Return
        "Accuracy": float
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, predictions, references) -> Dict[str, float]:
        if isinstance(self.tokenizer, tiktoken.Encoding):
            refs_len = [len(ref) for ref in self.tokenizer.encode_batch(references)]
            trunced_preds = [pred[:l] for l, pred in zip(refs_len, self.tokenizer.encode_batch(predictions))]
            trunced_preds = self.tokenizer.decode_batch(trunced_preds)
        else: # transformers
            refs_len = [len(ref) for ref in self.tokenizer(references, add_special_tokens=False)['input_ids']]
            trunced_preds = [pred[:l] for l, pred in zip(refs_len, self.tokenizer(predictions, add_special_tokens=False)['input_ids'])]
            trunced_preds = self.tokenizer.batch_decode(trunced_preds, clean_up_tokenization_spaces=False)

        trunced_preds = [p.strip() for p in trunced_preds]
        references = [r.strip() for r in references]
        score_list = np.asarray(trunced_preds) == np.asarray(references)
        self._last_score_lists = {'Accuracy': score_list}
        return {'Accuracy': np.mean(score_list) * 100}
