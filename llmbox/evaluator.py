from logging import getLogger
from statistics import mode
from typing import Dict, Tuple

from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import load_dataset
from .model import load_model
from .utils import DatasetArguments, EvaluationArguments, ModelArguments, catch_error, dynamic_stride_tqdm

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
        if self.model_args.vllm:
            from vllm import LLM

            if isinstance(self.model.model, LLM):
                self.dataset_args.batch_size = -1
                logger.info(
                    "Setting batch_size to -1, since vllm can automatically planning the optimal batch and order."
                )
        self.dataset = load_dataset(self.dataset_args, self.model)

    @catch_error
    def evaluate(self) -> Dict[str, float]:
        r"""It conducts the evaluation on the dataset with corresponding models.
        We support two evaluation types:

            - `Ranking`, ranking several options given a context, mainly applicable for multi-choice tasks. We compute the PPL scores of each option and select the one with lowest PPL.
            - `Generation`, generating the response based on the context, applicable for most of tasks. We directly call the `generation` interface of each model or API.

        Finally, we call the `calculate_metric` to get the metric score of prediction results.
        """
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.dataset_args.batch_size if self.dataset_args.batch_size != -1 else self.dataset.len(),
            collate_fn=lambda x: x,
            shuffle=False,
            pin_memory=True,
        )

        if self.dataset.evaluation_type == 'ranking':
            call_model = self.model.get_ppl
        elif self.dataset.evaluation_type == 'generation':
            call_model = self.model.generation
        elif self.dataset.evaluation_type == "user_defined":
            call_model = self.dataset.evaluation
        else:
            raise ValueError(
                f"We only support three evaluation types: `ranking`, `generation`, and `user_defined`, but got `{self.dataset.evaluation_type}`."
            )

        if self.evaluation_args.dry_run:
            if self.dataset.evaluation_type == "ranking":

                def call_model(batch):
                    return [(0, 1)] * len(batch)
            else:

                def call_model(batch):
                    return [""] * len(batch)

        # use tqdm for non-vllm models
        if self.dataset_args.batch_size != -1:
            tqdm_kwargs = dict(iterable=dataloader, desc=self.dataset.name, dynamic_ncols=True, unit="example")
            if self.dataset.evaluation_type == "ranking":
                # dataloader is often sacled by batch size and option nums, comparing to evaluation data
                stride_scale = self.dataset_args.batch_size
                if self.dataset.use_normalization:
                    stride_scale /= 2
                dataloader = dynamic_stride_tqdm(
                    strides=self.dataset.option_nums, stride_scale=stride_scale, **tqdm_kwargs
                )
            else:
                dataloader = tqdm(unit_scale=self.dataset_args.batch_size, **tqdm_kwargs)

        # call model
        raw_predictions = []
        for batch in dataloader:
            raw_predictions.extend(call_model(batch))
            self.dataset.log_predictions(raw_predictions)

        if len(raw_predictions) != self.dataset.len():
            raise RuntimeError(
                f"The number of results {len(raw_predictions)} should be equal to the number of samples in the dataset {self.dataset.len()}."
            )

        # post processing and self-consistency
        predictions = self.dataset.post_processing(raw_predictions)
        if len(predictions) != self.dataset.len(option_num=False, normalization=False):
            raise RuntimeError(
                f"The number of results {len(predictions)} should be equal to the number of samples in the dataset {self.dataset.len(option_num=False, normalization=False)}."
            )

        step = self.dataset.len(option_num=False, sample_num=False, normalization=False)
        mode_predictions = [mode(predictions[i::step]) for i in range(step)]

        # calculate metric
        metric_results, last_score_lists = self.dataset.calculate_metric(mode_predictions)
        self.dataset.log_predictions(raw_predictions, predictions, last_score_lists)

        msg = f"Evaluation finished successfully:"
        if not isinstance(next(iter(metric_results.values())), dict):
            metric_results = {self.dataset.name: metric_results}

        for dataset_name, result in metric_results.items():
            msg += f"\n##### {dataset_name} #####"
            for key, value in result.items():
                msg += "\n{}: {:.2f}".format(key, value)

        logger.info(msg)
        return metric_results
