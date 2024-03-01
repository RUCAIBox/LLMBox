import itertools
from typing import List, Union
import numpy as np
from tqdm import tqdm

from .metric import Metric
from ..dataset.gsm8k import Timeout


class PassAtK(Metric):
    r""" Calculate the Pass@K score.

    Return:
        "Pass@K": float

    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, predictions, references):
        result = []
        for samples, refer in tqdm(zip(predictions, references), desc="Evaluating Pass@K", total=len(predictions)):
            sample_result = []
            for pred in samples:
                # check_program = refer["prompt"] + pred + "\n" + refer["test"] + "\n" + f"check({refer['entry_point']})"
                check_program = refer.replace("{pred}", pred)
                with Timeout():
                    try:
                        exec(check_program)
                        sample_result.append('passed')
                    except TimeoutError:
                        sample_result.append("timed out")
                    except AssertionError:
                        sample_result.append(f"failed: AssertionError")
                    except BaseException as e:
                        sample_result.append(f"failed: {e}")
            result.append(sample_result)

        total, correct = [], []
        for sample_result in result:
            total.append(len(sample_result))
            correct.append(sample_result.count('passed'))
        pass_at_k = self.estimate_pass_at_k(total, correct, self.k)
        self._last_score_lists = {f"pass@{self.k}": pass_at_k}
        return {f"pass@{self.k}": np.mean(pass_at_k)}

    @staticmethod
    def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray], num_correct: Union[List[int], np.ndarray], k: int
    ) -> np.ndarray:
        """
        Estimates pass@k of each problem and returns them in an array.
        """

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
