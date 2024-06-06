# How to Load Datasets from Hugging Face

In this tutorial, we will learn how to download datasets from Hugging Face using the [`datasets`](https://huggingface.co/docs/datasets/en/index) library. The `datasets` library is a powerful tool that allows you to easily download and work with datasets from [Hugging Face](https://huggingface.co/datasets).

See a full list of supported datasets at [here](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/supported-datasets.md).

## Case 1: Directly load from Hugging Face

By default, `LLMBox` will handle everything for you. You just need to specify the dataset name in the command line.

```python
python inference.py -m model -d mmlu
```

The dataset will be downloaded and cached in the `~/.cache/huggingface/datasets` directory.

## Case 2: Load from a Hugging Face mirror

Datasets

To load a dataset from a Hugging Face mirror, you can use the `--hf_mirror` flag. The dataset will be downloaded from Hugging Face mirror using `hfd.sh`.

This is an experimental feature and may not work in some environments. If you encounter any issues, please let us know.

```shell
python inference.py -m model -d mmlu --hf_mirror
```

`hfd.sh` is a slightly modified version of `huggingface-cli` download [wrapper](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/), which offers a more stable and faster download speed than the original `huggingface-cli`.

`hfd.sh` will download the dataset from the Hugging Face mirror and cache it in the `~/.cache/huggingface/datasets` directory. Then `datasets` will load the dataset from the cache.

The next time you run the command, `datasets` will directly load the dataset from the cache:

```shell
python inference.py -m another-model -d mmlu
```

## Case 3: Load local dataset in offline mode

If you have already downloaded the dataset and want to load it in offline mode, you can use `--dataset_path` to specify the dataset path.

```shell
python inference.py -m model -d mmlu --dataset_path path/to/mmlu
```

The dataset will be loaded from the specified path.


```bash
# from a cloned directory of the huggingface dataset repository:
python inference.py -d copa --dataset_path /path/to/copa

# from a local (nested) directory saved by `dataset.save_to_disk`:
python inference.py -d race --dataset_path /path/to/race/middle
python inference.py -d race:middle --dataset_path /path/to/race
python inference.py -d race:middle --dataset_path /path/to/race/middle
python inference.py -d race:middle,high --dataset_path /path/to/race
```

`dataset_path` can also accept a dataset file or a directory containing these files (supports json, jsonl, csv, and txt):
```bash
# load one split from one subset only
python inference.py -d gsm8k --dataset_path /path/to/gsm.jsonl
python inference.py -d race --dataset_path /path/to/race/middle/train.json

# load test and train splits from middle subset (a directory contains `/path/to/race/middle/train.json` and `/path/to/race/middle/test.json`)
python inference.py -d race --dataset_path /path/to/race/middle --evaluation_set "test[:10]" --example_set "train"

# load test and train splits from middle and high subsets (a nested directory)
python inference.py -d race:middle,high --dataset_path /path/to/race --evaluation_set "test[:10]" --example_set "train"

# load test and train splits from middle and high subsets with a filename pattern
python inference.py -d race:middle,high --evaluation_set "test[:10]" --example_set "train" --dataset_path "/pattern/of/race_{subset}_{split}.json"
python inference.py -d mmlu --evaluation_set val --example_set dev --dataset_path "/pattern/of/mmlu/{split}/{subset}_{split}.csv"
```
