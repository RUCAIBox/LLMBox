import numpy as np
from torch.utils.data import DataLoader
from model import load_model
from dataset import load_dataset
from tqdm import tqdm


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
        self.args = args

        self.model = load_model(args)
        self.dataset = load_dataset(args, self.model)
        # TODO: change to logger
        # filename = args.model + "-" + args.dataset + "-" + str(args.num_shots)
        # self.args.filename = filename

    def evaluate(self):
        r"""It conducts the evaluation on the dataset with corresponding models.
        We support two evaluation types:

            - `Ranking`, ranking several options given a context, mainly applicable for multi-choice tasks. We compute the PPL scores of each option and select the one with lowest PPL.
            - `Generation`, generating the response based on the context, applicable for most of tasks. We directly call the `generation` interface of each model or API.
        
        Finally, we call the `calcuate_metric` to get the metric score of prediction results.
        """
        dataloader = DataLoader(
            self.dataset, batch_size=self.args.batch_size, collate_fn=lambda x: x, shuffle=False, pin_memory=True
        )

        results = []
        for batch in tqdm(dataloader, dynamic_ncols=True, desc="Evaluating"):
            if self.dataset.evaluation_type == 'ranking':
                results.extend(self.model.get_ppl(batch))
            elif self.dataset.evaluation_type == 'generation':
                results.extend(self.model.generation(batch))
            else:
                raise ValueError(f"We only support two evaluation types: `ranking` and `generation`.")
        assert len(results) == len(self.dataset)

        # if self.dataset.evaluation_type == 'ranking':
        #     labels = []
        #     st = 0
        #     results = np.array(results)
        #     for num in self.dataset.option_nums:
        #         labels.append(results[st:st + num].argmin())
        #         st += num
        #     results = labels
        #     assert len(results) == len(self.dataset.references)

        print('#' * 5, self.dataset.name, '#' * 5)
        scores = self.dataset.calculate_metric(results)
        for key, value in scores.items():
            print("{}: {:.2f}".format(key, value))
