<!-- ## Supervised Fine-tuning and Continual Pre-training -->
# LLMBox-Training Tools

<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ymcui/Chinese-LLaMA-Alpaca">
    <a href="https://app.codacy.com/gh/ymcui/Chinese-LLaMA-Alpaca/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde"/></a>
</p>

## Features & Functions

大致具有的功能介绍，分点列出

## Overview
整体框架，放论文中的图

论文中对图的介绍

The file directory inside the ZIP file is as follows (using Chinese-LLaMA as an example):

```

  - dpo.py                   # DPO training file
  - evol_instruct.py         # Evol_instruct file
  - gptq.py                  # GPTQ training file
  - merge_tokenizer.py       # tokenizer merging file (Before pre-training)
  - ppo.py                   # PPO training file
  - train.py                 # Training file with Pre-training and Supervised Fine-tuning codes
```

然后可以在这里添加表格，介绍已经有的数据集和模型方式等。

## Quick Start
With the source code, you can use multiple functions with following steps.

### 1. Supervised fine-tuning (instruction tuning)
#### Downloading the SFT dataset
```shell
bash download.sh
```

#### Training a SFT model
Just change the `data_path` in `bash/run_7b_ds3.sh` to the path you want, e.g., `data/alpaca_data.json`. Then run the following script:
```shell
bash bash/run_7b_ds3.sh
```

The default behaviour is to train the LLaMA-2-7b model on the subset of the alpaca dataset.

### 2. Continual pre-training with your own corpora

#### Merging tokenizer

If you want to add new tokens (such as Chinese characters) to your vocabulary and then continual pre-train the model on your corpora, you just need to prepare the corpora under the folder `data/chinese.txt` and run the following script:

```shell
bash bash/run_7b_pt.sh
```

It will first merge the vocabulary of your corpora and the original vocabulary and then tune the parameters of the whole model to adapt to your corpora.

#### User-defined symbols

If you want to add user-defined symbols when merging new tokenizers, you can rewrite the `user_defined_symbols.json`. 

```json
{
    "list": [
    	"symbol1",
    	"symbol2"
    ]
}
```

#### Others
You can also leverage part of the script in `bash/run_7b_pt.sh` to just merge the tokenizer or continual pre-train the model using your corpora with the original tokenizer.

### 3. Merging different dataset with designated ratios for training
LLMBox supports merging different datas together to train the model. Just change the `data_set` in `bash/run_7b_hybrid.sh` to a list of file names in `data_path` separated by white space. LLMBox will automatically concates the dataset together if you do not set `dataset_ratio` and `max_steps`.

If you want to merge different datas with designated ratios, you can change `dataset_ratio` to a list of floats separated by white space, and `max_steps` is required when setting `dataset_ratio`. LLMBox will merge the dataset with the ratio you set. After changing the parameters, you can run the following script:

```shell
bash bash/run_7b_hybrid.sh
```
The default behaviour is to train the LLaMA-2-7b model on the merged dataset.

### 4. Training with PPO(Proximal Policy Optimization)
If you want to use your individual data to train language models with PPO(Proximal Policy Optimization), you can change the `data_path` in `bash/run_ppo.sh` to the path you want, e.g., `data/ppo.json`. Then run the following script:
```shell
bash bash/run_ppo.sh
```
Since PPO training strategy requires a reward model, you can change the `reward_model_name` in `bash/run_ppo.sh` to the model path you want or the model in huggingface, e.g., `OpenAssistant/reward-model-deberta-v3-large-v2`. The default behaviour is to train the LLaMA-2-7b model on the dataset.

### 5. Training with DPO(Direct Preference Optimization)
If you want to use your data to train language models with DPO(Direct Preference Optimization), you can change the `data_path` in `bash/run_dpo.sh` to the path you want, e.g., `data/dpo.json`, or the dataset name in huggingface, e.g., `Dahoas/synthetic-instruct-gptj-pairwise`, which contains chosen and rejected texts for each query. Then run the following script:
```shell
bash bash/run_dpo.sh
```

The default behaviour is to train the LLaMA-2-7b model on the dataset.

### 6. Training with GPTQ(Generative Pre-trained Transformer for Querying Tables)
LLMBox supports training models with GPTQ. You can run the following script:
```shell
bash bash/run_gptq.sh
```
You can change the following parameters in `bash/run_gptq.sh` to adjust the training process, including `bits`,`group_size`,`damp_percent`,`use_triton`, and `unquantized_model_dtype`.
The default behaviour is to train the LLaMA-2-7b model with GPTQ.

### 7. Parameter Efficient Fine-tuning(LoRA and QLoRA)
LLMBox supports PEFT(Parameter Efficient Fine-tuning) strategies, including LoRA and QLoRA. Just add the following scripts in your bash files to enable LoRA or QLoRA.
```shell
    --lora True\
```
```shell
    --qlora True\
```
You can also simply run the following script:
```shell
bash bash/run_7b_lora.sh
```
```shell
bash bash/run_7b_qlora.sh
```
The default behaviour is to train the LLaMA-2-7b model on the subset of the alpaca dataset with LoRA or QLoRA.

### 8. Deepspeed?

## System Performance

给出一部分文章中的测试结果