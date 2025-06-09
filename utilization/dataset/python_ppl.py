from typing import Optional
from ..metric import PPL
from .validation_perplexity_dataset import ValidationPerplexityDataset


class PythonPPL(ValidationPerplexityDataset):
    r"""The dataset of python ppl.
    """

    evaluation_set = "train"
    load_args = ("vwxyzjn/the-algorithm-python", )
    metrics = [PPL()]

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, example_set)
        self.evaluation_data = [{"text": o["reference_solution"].strip()} for o in self.evaluation_data if len(o["reference_solution"].strip()) > 0]

    def format_instance(self, instance):
        return {
            "text": instance["text"],
            "options": [instance["text"]],
        }

