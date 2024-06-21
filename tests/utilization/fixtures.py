import sys
from typing import Dict, List, Optional

import pytest

sys.path.append('.')
from utilization.dataset.dataset import DatasetCollection
from utilization.dataset.generation_dataset import GenerationDataset
from utilization.load_dataset import load_dataset
from utilization.model.model_utils.conversation import Conversation
from utilization.model.openai_model import Openai
from utilization.utils.arguments import DatasetArguments, EvaluationArguments, ModelArguments
from utilization.utils.dynamic_stride_tqdm import dynamic_stride_tqdm
from utilization.utils.log_results import PredictionWriter


@pytest.fixture
def conversation():
    return Conversation(
        messages=[{
            "role": "system",
            "content": "This is a system message."
        }, {
            "role": "user",
            "content": "This is a user message."
        }, {
            "role": "assistant",
            "content": "This is an assistant message."
        }, {
            "role": "user",
            "content": "This is the second user message."
        }, {
            "role": "assistant",
            "content": "This is the second assistant message."
        }]
    )


@pytest.fixture
def get_openai_model():

    def openai_model(chat_template, model_backend="huggingface"):
        args = ModelArguments(
            model_name_or_path="fake_model",
            model_type="chat",
            tokenizer_name_or_path="cl100k_base",
            model_backend="openai",
            openai_api_key="fake_key",
            chat_template=chat_template,
            api_endpoint="completions"
        )
        model = Openai(args)
        args.model_backend = model_backend
        model.model_backend = model_backend

        model.get_ppl = lambda x: [(0, 1)] * len(x)
        model.generation = lambda x: [""] * len(x)
        model.get_prob = lambda x: [[1 / p[1]] * p[1] for p in x]

        return model

    return openai_model


@pytest.fixture
def get_dataset(get_openai_model):

    def dataset(
        dataset_name: str,
        num_shots: int,
        cot: Optional[str] = None,
        chat_template: Optional[str] = None,
        ranking_type: Optional[str] = None,
        batch_size: int = 1,
        model_backend = "huggingface",
        **kwargs
    ):

        args = DatasetArguments(
            dataset_names=[dataset_name],
            num_shots=num_shots,
            batch_size=batch_size,
            cot=cot,
            ranking_type=ranking_type,
            **kwargs,
        )
        evaluation_args = EvaluationArguments()
        openai_model = get_openai_model(chat_template, model_backend)

        ds = list(load_dataset(dataset_name, args, openai_model, evaluation_args))

        return ds[0][dataset_name]

    return dataset


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
