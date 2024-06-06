# How to Load Datasets with Subsets

Some datasets have multiple subsets. For example, Massive Multitask Language Understanding (`mmlu`) dataset contains 57 different subsets categorized into four categories: `stem`, `social_sciences`, `humanities`, and `other`.

While some other dataset is a subset of another dataset. For example, Choice Of Plausible Alternatives (`copa`) is a subset of `super_glue`.

See a full list of supported datasets at [here](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/supported-datasets.md).

## Load from huggingface server

We use the `datasets` library to load the dataset from the huggingface server. If you have issue connecting to the Internet or the Hugging Face server, see [here](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/how-to-load-datasets-from-huggingface.md) for help.

Load a dataset that is a subset of another dataset (e.g. `copa`):

```shell
python inference.py -d copa
```

Load a dataset with multiple subsets (e.g. `mmlu`):

```shell
python inference.py -d mmlu:abstract_algebra,human_sexuality
```

In some cases, you may want to load a specific split of the dataset (e.g. `test`, `dev`, `validation`, ...). Both `evaluation_set` and `example_set` support the Huggingface [String API](https://huggingface.co/docs/datasets/loading#slice-splits):

```shell
python inference.py -d race:middle,high --evaluation_set "test[:10]" --example_set "train"
```

## Understand the behaviour of subsets

By default we load all the subsets of a dataset:

```shell
python inference.py -m model -d mmlu
# expands to all 57 subsets
# equivalent: mmlu:abstract_algebra,human_sexuality,human_sexuality,...
# equivalent: mmlu:[stem],[social_sciences],[humanities],[other]
```

```shell
python inference.py -m model -d arc
# equivalent: arc:ARC-Easy,ARC-Challenge
```

Unless a default subset is defined (see [supported datsaets](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/supported-datasets.md) for all the default subsets):

```bash
python inference.py -m model -d cnn_dailymail
# equivalent: cnn_dailymail:3.0.0
```

Some datasets like GPQA (Google-Proof Q&A) have to load example set separately. You need to download the dataset to any directory and provide the path to the dataset:

```bash
# few_shot
python inference.py -m model -d gpqa --ranking_type generation -shots 5 --example_set "../gpqa/prompts"
```

## Overriding `load_raw_dataset` function

Also feel free to override this function if you want to load the dataset in a different way:

```python
from .utils import load_raw_dataset_from_file, get_raw_dataset_loader

class MyDataset(Dataset):
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        self.evaluation_data = get_raw_dataset_loader(...)("test")
        self.example_data = load_raw_dataset_from_file("examples.json")
```

For more details on how to customize the dataset, see this [guide](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/how-to-customize-dataset.md).
