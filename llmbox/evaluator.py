from logging import getLogger
from statistics import mode
from typing import Dict, Tuple

from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from .dataset import load_dataset
from .model import load_model
from .utils import DatasetArguments, EvaluationArguments, ModelArguments, catch_error, dynamic_interval_tqdm

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
            batch_size=self.dataset_args.batch_size if self.dataset_args.batch_size != -1 else len(self.dataset),
            collate_fn=lambda x: x,
            shuffle=False,
            pin_memory=True
        )

        if self.dataset.evaluation_type == 'ranking':
            self.model.set_ppl_args(**self.dataset.model_args)
            call_model = self.model.get_ppl
        elif self.dataset.evaluation_type == 'generation':
            self.model.set_generation_args(**self.dataset.model_args)
            call_model = self.model.generation
        elif self.dataset.evaluation_type == 'user_defined':
            call_model = self.dataset.evaluation
        else:
            raise ValueError(
                f"We only support three evaluation types: `ranking`, `generation`, and `user_defined`, but got `{self.dataset.evaluation_type}`."
            )

        # call model
        raw_predictions = []
        if self.dataset_args.batch_size != -1:
            dataloader = dynamic_interval_tqdm(
                iterable=dataloader,
                intervals=self.dataset.option_nums,
                desc=self.dataset.name,
                dynamic_ncols=True,
                total=len(self.dataset.evaluation_data),
                unit="example",
            )
        for batch in dataloader:
            raw_predictions.extend(call_model(batch))
            self.dataset.log_predictions(raw_predictions)

        if len(raw_predictions) != len(self.dataset):
            raise RuntimeError("The number of results should be equal to the number of samples in the dataset.")

        # post processing and self-consistency
        predictions = self.dataset.post_processing(raw_predictions)
        assert len(predictions) == len(self.dataset.references) * self.dataset_args.sample_num
        self.dataset.log_predictions(raw_predictions, predictions)

        step = len(predictions) // self.dataset_args.sample_num
        mode_results = [mode(predictions[i::step]) for i in range(step)]

        # calculate metric
        metric_results = self.dataset.calculate_metric(mode_results)

        msg = f'Evaluation finished successfully:'
        if not isinstance(next(iter(metric_results.values())), dict):
            metric_results = {self.dataset.name: metric_results}

        for dataset_name, result in metric_results.items():
            msg += f'\n##### {dataset_name} #####'
            for key, value in result.items():
                msg += "\n{}: {:.2f}".format(key, value)

        logger.info(msg)
        return metric_results
