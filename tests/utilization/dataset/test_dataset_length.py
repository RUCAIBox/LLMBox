import io
import json

import pytest
from torch.utils.data import DataLoader

from ..fixtures import DatasetCollection, PredictionWriter, dynamic_stride_tqdm, get_dataset, get_openai_model

datasets = [
    ("mmlu:abstract_algebra", 4, "ppl_no_option"),
    ("mmlu:abstract_algebra", 4, "ppl"),
    ("mmlu:abstract_algebra", 1, "prob"),
    ("openbookqa", 4, "ppl_no_option"),  # acc:norm
    ("gsm8k", 1, "generation"),
]


class FakeWriter(PredictionWriter):

    def __init__(self, path):
        super().__init__(path)
        self.f = io.StringIO()

    def _write(self, data):
        json.dump(data, self.f, ensure_ascii=False)
        self.f.write("\n")


@pytest.mark.parametrize("dataset, num_options, ranking_type", datasets)
def test_dataset_length(get_dataset, dataset, num_options, ranking_type):
    d = get_dataset(dataset, 0, batch_size=16, model_backend="openai", ranking_type=ranking_type)
    dataset = DatasetCollection({dataset: d})
    assert len(list(dataset._lines_iter)) == len(list(dataset))

    dataset = DatasetCollection({dataset: d})
    num_instances = len(list(dataset))
    num_questions = len(dataset.evaluation_data)
    assert num_instances == num_questions * num_options * (2 if dataset.name == "openbookqa" else 1)
    assert num_instances == sum(dataset.strides)
    assert num_questions == len(dataset.strides)
    assert dataset._lines_iter

    batch_sampler = dataset.get_batch_sampler(False)
    dataloader = DataLoader(
        dataset,
        collate_fn=lambda x: x,
        pin_memory=True,
        num_workers=0,
        batch_sampler=batch_sampler,
    )

    num_batches = len(list(dataloader))
    assert num_batches == (num_instances + 15) // 16

    dataloader = dynamic_stride_tqdm(
        dataloader,
        strides=dataset.strides,
        desc=dataset.name,
        dynamic_ncols=True,
        unit=" instances",
        continue_from=dataset.args.continue_from,
    )

    writer = FakeWriter("")
    raw_predictions = []
    tqdm_batches = list(dataloader)
    for batch in tqdm_batches:
        batch_results = batch_sampler.call_model(batch)
        raw_predictions.extend(batch_results)
        dataset.step(writer, dataloader, batch_results)

    assert len(tqdm_batches) == num_batches
    assert sum(len(batch) for batch in tqdm_batches) == num_instances
    assert len(raw_predictions) == num_instances
    assert len(writer.f.getvalue().splitlines()) == num_instances
