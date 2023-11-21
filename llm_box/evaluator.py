from logging import getLogger

import numpy as np
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import load_dataset
from .model import load_model

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

    def __init__(self, args):
        model_args, dataset_args, evaluation_args = args
        self.model_args = model_args
        self.dataset_args = dataset_args
        self.evaluation_args = evaluation_args

        set_seed(self.evaluation_args.seed)

        self.model_args.sample_path = self.evaluation_args.sample_path
        self.model = load_model(self.model_args)
        self.dataset = load_dataset(self.dataset_args, self.model)
        # TODO: change to logger
        # filename = args.model + "-" + args.dataset + "-" + str(args.num_shots)
        # self.args.filename = filename

    def evaluate(self):
        r"""It conducts the evaluation on the dataset with corresponding models.
        We support two evaluation types:

            - `Ranking`, ranking several options given a context, mainly applicable for multi-choice tasks. We compute the PPL scores of each option and select the one with lowest PPL.
            - `Generation`, generating the response based on the context, applicable for most of tasks. We directly call the `generation` interface of each model or API.

        Finally, we call the `calculate_metric` to get the metric score of prediction results.
        """
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.dataset_args.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
            pin_memory=True
        )

        results = []
        for batch in tqdm(dataloader, dynamic_ncols=True, desc="Evaluating"):
            if self.dataset.evaluation_type == 'ranking':
                batch_results = []
                for _ in range(self.evaluation_args.sample_path):
                    batch_results.append(self.model.get_ppl(batch))
                batch_results = list(map(list, zip(*batch_results)))  # transpose
                results.extend(batch_results)
            elif self.dataset.evaluation_type == 'generation':
                batch_results = []
                for _ in range(self.evaluation_args.sample_path):
                    batch_results.append(self.model.generation(batch))
                batch_results = list(map(list, zip(*batch_results)))  # transpose
                results.extend(batch_results)
            else:
                raise ValueError(f"We only support two evaluation types: `ranking` and `generation`.")

        assert len(results) == len(self.dataset)

        print('#' * 5, self.dataset.name, '#' * 5)
        scores = self.dataset.calculate_metric(results)
        for key, value in scores.items():
            print("{}: {:.2f}".format(key, value))
