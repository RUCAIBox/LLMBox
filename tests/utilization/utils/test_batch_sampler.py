import sys

from ..fixtures import get_dataset_collection

sys.path.append('.')
from utilization.model.huggingface_model import HuggingFaceModel
from utilization.model.model_utils.batch_sampler import AutoBatchSizeSampler, DatasetCollectionBatchSampler
from utilization.utils.arguments import ModelArguments


def test_auto_batch_sampler_auto_batching():
    data = ["This is a long text", "short one", "short one"]
    bs = AutoBatchSizeSampler(data, 1, True)
    assert bs.data_order == [[0], [1, 2]]


def test_auto_batch_sampler():
    data = ["This is a long text", "short one", "short one"]
    bs = AutoBatchSizeSampler(data, 1, False)
    assert bs.data_order == [[0], [1], [2]]


def test_dcbs(get_dataset_collection):
    data = get_dataset_collection({
        "test1": ["This is a long text", "short one", "short one"],
        "test2": ["这是一段长文本", "短文本", "短文本"]
    })
    bs = DatasetCollectionBatchSampler(data, 1)
    assert list(iter(bs)) == [[0], [1], [2], [3], [4], [5]]


def test_dcbs_few_shot_prefix_caching(get_dataset_collection):
    model_args = ModelArguments(model_name_or_path="gpt2", model_type="base", prefix_caching=True)
    model = HuggingFaceModel(model_args)
    data = get_dataset_collection(
        evaluation_datasets={
            "test1": ["This is a long text", "short one", "short one"],
            "test2": ["这是一段用中文书写的长文本", "短文本", "短文本"]
        },
        example_datasets=[["This is a few-shot example", "example"], ["这是一个小样本", "样本"]],
        prefix_caching=True,
        batch_size=3,
        model=model,
    )
    bs = DatasetCollectionBatchSampler(data, 3)

    assert list(iter(bs)) == [[0, 3], [0, 1, 2, 3], [4, 5]]
    assert data.model.use_cache == True


def test_dcbs_few_shot(get_dataset_collection):
    data = get_dataset_collection(
        evaluation_datasets={
            "test1": ["This is a long text", "short one", "short one"],
            "test2": ["这是一段用中文书写的长文本", "短文本", "短文本"]
        },
        example_datasets=[["This is a few-shot example", "example"], ["这是一个小样本", "样本"]],
        prefix_caching=True,
        batch_size=3,
    )
    bs = DatasetCollectionBatchSampler(data, 3)

    assert list(iter(bs)) == [[0, 1, 2], [3, 4, 5]]
    assert data.model.support_cache == False


def test_dcbs_auto_batching(get_dataset_collection):
    data = get_dataset_collection({
        "test1": ["This is a long text", "short one", "short one"],
        "test2": ["这是一段用中文书写的长文本", "短文本", "短文本"]
    })
    bs = DatasetCollectionBatchSampler(data, 1, auto_batch_size=True)

    assert list(iter(bs)) == [[0], [1, 2], [3], [4, 5]]
