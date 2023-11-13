from logging import getLogger
from time import perf_counter
from typing import Any, Dict, Tuple

from accelerate.utils import set_seed
from tqdm import tqdm

from .dataset import load_dataset
from .dataset.dataset import Dataset as LLMDataset
from .model import load_model
from .model.model import Model
from .utils import ModelArguments, DatasetArguments, EvaluationArguments

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

    def __init__(self, args: Tuple[ModelArguments, DatasetArguments, EvaluationArguments]):
        model_args, dataset_args, evaluation_args = args
        self.model_args = model_args
        self.dataset_args = dataset_args
        self.evaluation_args = evaluation_args

        set_seed(self.evaluation_args.seed)

        self.model = load_model(self.model_args)
        self.loaded_datasets = []
        for dataset_name, subset in self.dataset_args.parse_dataset_names():
            loaded_dataset = load_dataset(self.dataset_args, self.model, dataset_name, subset)
            self.loaded_datasets.extend(loaded_dataset)

    def evaluate(self) -> dict:
        r"""It conducts the evaluation on the dataset with corresponding models.
        We support two evaluation types:

            - `Ranking`, ranking several options given a context, mainly applicable for multi-choice tasks. We compute the PPL scores of each option and select the one with lowest PPL.
            - `Generation`, generating the response based on the context, applicable for most of tasks. We directly call the `generation` interface of each model or API.

        Finally, we call the `calcuate_metric` to get the metric score of prediction results.
        """
        logger.info(f"Start evaluating {self.model.name} on {self.dataset_args.datasets}.")
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
        logger.debug(f"The number of samples in the dataset: {len(dataloader)}")
        logger.debug(f"The first batch of the dataset: {next(iter(dataloader))}")

        # call model
        start_time = perf_counter()
        predictions = []
        for batch in tqdm(dataloader, dynamic_ncols=True, desc="Evaluating"):
            predictions.extend(call_model(batch))
        end_time = perf_counter()
        pref_results = Evaluator._compute_time_perf(start_time, end_time, len(predictions))

        if len(predictions) != len(dataloader):
            raise RuntimeError("The number of results should be equal to the number of samples in the dataset.")

        metric_results = dataset.calculate_metric(predictions)
        metric_results.update(pref_results)

        print('#' * 5, dataset.name, '#' * 5)
        for key, value in metric_results.items():
            print("{}: {:.2f}".format(key, value))
        return metric_results

    @staticmethod
    def _compute_time_perf(start_time: float, end_time: float, num_samples: int) -> Dict[str, Any]:
        """
        A utility function computing time performance metrics:
            - `total_time_in_seconds` - pipeline inference runtime for the evaluation data in seconds,
            - `samples_per_second` - pipeline throughput in the number of samples per second.
            - `latency_in_seconds` - pipeline inference runtime for the evaluation data in seconds per sample,

        Reference: https://github.com/huggingface/evaluate/blob/ec46ca2cbb77433622057a74cabe611defae669d/src/evaluate/evaluator/base.py#L163
        """
        latency = end_time - start_time
        throughput = num_samples / latency
        latency_sample = 1.0 / throughput

        return {
            "total_time_in_seconds": latency,
            "samples_per_second": throughput,
            "latency_in_seconds": latency_sample,
        }
