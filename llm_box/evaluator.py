import threading
from argparse import Namespace
from logging import getLogger
from time import perf_counter
from typing import Any, Dict, Tuple, NoReturn

import numpy as np
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import load_dataset
from .dataset.dataset import Dataset as LLMDataset
from .model import load_model, load_tokenizer
from .model.model import Model
from .utils import args_to_tokenizer_kwargs

logger = getLogger(__name__)


class Evaluator:
    r"""The class for the evaluation pipeline.
    It loads the model and dataset, and then conducts evaluation.

    Args:
        args (Namespace): The global configurations.

    Attributes:
        model (Model): Our class for model.
        dataset (Dataset): Our class for dataset.
    """

    def __init__(self, args: Namespace):
        self.args = args
        set_seed(args.seed)

        tokenizer_kwargs = args_to_tokenizer_kwargs(args)
        args.tokenizer = load_tokenizer(
            args.model,
            # set `pad` token to `eos` token if pad token is not set
            # (as is the case for llama models)
            {'pad': 'eos'},
            tokenizer_kwargs,
        )
        self.model = None
        self.loaded_datasets = None
        self._load_model_and_dataset()

        # TODO: change to logger
        # filename = args.model + "-" + args.dataset + "-" + str(args.num_shots)
        # self.args.filename = filename

    def _load_model_and_dataset(self, multi_thread: bool = True):
        if multi_thread:
            model_thread = threading.Thread(target=self._load_model)
            dataset_thread = threading.Thread(target=self._load_dataset)

            dataset_thread.start()
            model_thread.start()

            model_thread.join()
            dataset_thread.join()
        else:
            self._load_model()
            self._load_dataset()

    def _load_model(self) -> NoReturn:
        self.model = load_model(self.args)
        print("==>", self.model)

    def _load_dataset(self) -> NoReturn:
        results = []
        for dataset_name in self.args.dataset.split(","):
            loaded_dataset = load_dataset(self.args, *dataset_name.split(":"))
            results.extend(loaded_dataset)
        self.loaded_datasets = results
        print("==>", self.args.dataset.split(","), self.loaded_datasets)

    def evaluate(self) -> dict:
        r"""It conducts the evaluation on the dataset with corresponding models.
        We support two evaluation types:

            - `Ranking`, ranking several options given a context, mainly applicable for multi-choice tasks. We compute the PPL scores of each option and select the one with lowest PPL.
            - `Generation`, generating the response based on the context, applicable for most of tasks. We directly call the `generation` interface of each model or API.

        Finally, we call the `calcuate_metric` to get the metric score of prediction results.
        """
        results = {}
        for dataset in self.loaded_datasets:
            results[dataset.name] = self._evaluate_once(self.model, dataset)
        return results

    @staticmethod
    def _evaluate_once(
        model: Model,
        dataset: LLMDataset,
    ) -> dict:

        if dataset.evaluation_type == 'ranking':
            call_model = model.get_ppl
        elif dataset.evaluation_type == 'generation':
            call_model = model.generation
        else:
            raise ValueError(f"We only support two evaluation types: `ranking` and `generation`.")

        dataloader = dataset.get_dataloader()

        predictions, pref_results = Evaluator._call_model(call_model, dataloader)
        metric_inputs = Evaluator._predictions_processor(dataset.evaluation_type, predictions)

        metric_results = dataset.calculate_metrics(metric_inputs)
        metric_results.update(pref_results)

        Evaluator._summarize(dataset.name, metric_results)
        return metric_results

    @staticmethod
    def _predictions_processor(dataset: LLMDataset, evaluation_type: str, predictions: list) -> Dict:
        if evaluation_type == 'ranking':
            results = []
            st = 0
            predictions = np.array(predictions)
            for num in dataset.option_nums:
                results.append(predictions[st:st + num].argmin())
                st += num
            assert len(results) == len(dataset.references)
            return results

    @staticmethod
    def _call_model(call_model: callable, dataloader: DataLoader) -> Tuple[list, Dict[str, float]]:
        start_time = perf_counter()
        results = []
        for batch in tqdm(dataloader, dynamic_ncols=True, desc="Evaluating"):
            results.extend(call_model(batch))
        end_time = perf_counter()

        if len(results) != len(dataloader):
            raise RuntimeError("The number of results should be equal to the number of samples in the dataset.")
        return results, Evaluator._compute_time_perf(start_time, end_time, len(results))

    @staticmethod
    def _summarize(desc: str, metric_results: Dict[str, float]) -> None:
        """Print the evaluation results."""
        print('#' * 5, desc, '#' * 5)
        for key, value in metric_results.items():
            print("{}: {:.2f}".format(key, float(value)))

    @staticmethod
    def _compute_time_perf(start_time: float, end_time: float, num_samples: int) -> Dict[str, Any]:
        """
        A utility function computing time performance metrics:
            - `total_time_in_seconds` - pipeline inference runtime for the evaluation data in seconds,
            - `samples_per_second` - pipeline throughput in the number of samples per second.
            - `latency_in_seconds` - pipeline inference runtime for the evaluation data in seconds per sample,

        """
        latency = end_time - start_time
        throughput = num_samples / latency
        latency_sample = 1.0 / throughput

        return {
            "total_time_in_seconds": latency,
            "samples_per_second": throughput,
            "latency_in_seconds": latency_sample,
        }
