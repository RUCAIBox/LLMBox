import sys
from typing import Dict, List

import pytest

sys.path.append('.')
from utilization.dataset.dataset import DatasetCollection
from utilization.dataset.generation_dataset import GenerationDataset
from utilization.model.openai_model import Openai
from utilization.utils.arguments import DatasetArguments, ModelArguments


class FakeDataset(GenerationDataset):

    instruction = "{source}"

    def format_instance(self, instance: str) -> dict:
        return {"source": instance, "target": "tgt_of(" + instance + ")"}

    @property
    def refernces(self):
        return [None] * len(self.evaluation_data)


@pytest.fixture
def get_dataset_collection():

    def dataset_collection(
        evaluation_datasets: Dict[str, List[str]],
        example_datasets: List[List[str]] = None,
        prefix_caching: bool = False,
        batch_size: int = 1,
        model=None,
    ):

        if example_datasets is not None:
            assert len(evaluation_datasets) == len(example_datasets)
            max_num_shots = max(len(example) for example in example_datasets)
        else:
            example_datasets = [None] * len(evaluation_datasets)
            max_num_shots = 0

        dataset_args = DatasetArguments(
            dataset_names=list(evaluation_datasets.keys()), num_shots=max_num_shots, batch_size=batch_size
        )
        if model is None:
            model_args = ModelArguments(
                model_name_or_path="gpt2",
                model_type="base",
                openai_api_key="fake_api_key",
            )
            model = Openai(model_args)

        datasets = {}

        for (dataset_name, data), examples in zip(evaluation_datasets.items(), example_datasets):
            print(examples, max_num_shots)
            datasets[dataset_name] = FakeDataset(
                dataset_name, dataset_args, model, evaluation_data=data, example_data=examples
            )
        return DatasetCollection(datasets)

    return dataset_collection
