import itertools
from typing import List, Union
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Manager
import concurrent.futures
import os

from .metric import Metric


class PassAtK(Metric):
    r""" Calculate the Pass@K score.

    Return:
        "Pass@K": float

    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, predictions, references):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        def task_func(sample_tuple, _self):
            samples, refer = sample_tuple
            sample_result = []
            for pred in samples:
                check_program = refer.replace("{pred}", pred)
                res = _self.run_code_with_timeout(check_program)
                sample_result.append(res)
            return sample_result

        results = []

        with tqdm(total=len(predictions), desc="Evaluating Pass@K") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                for result in executor.map(task_func, zip(predictions, references), itertools.repeat(self)):
                    results.append(result)
                    pbar.update()

        total, correct = [], []
        for sample_result in results:
            total.append(len(sample_result))
            correct.append(sample_result.count('passed'))
        pass_at_k = self.estimate_pass_at_k(total, correct, self.k)
        self._last_score_lists = {f"pass@{self.k}": pass_at_k}
        return {f"pass@{self.k}": np.mean(pass_at_k)}

    def run_code_with_timeout(self, code_string, timeout=1):
        with Manager() as manager:
            result_dict = manager.dict()
            process = Process(target=self.exec_code, args=(code_string, result_dict))
            process.start()
            process.join(timeout=timeout)
            if process.is_alive():
                process.kill()
                return "timeout"
            else:
                return result_dict['result']

    @staticmethod
    def exec_code(code, result_dict):
        result_dict['result'] = 'Not executed'
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result_dict['result'] = 'passed'
        except AssertionError:
            result_dict['result'] = 'Assertion Error'
        except Exception as e:
            result_dict['result'] = f'Error: {str(e)}'

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
