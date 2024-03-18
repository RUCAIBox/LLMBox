# LLMBox

## Key Features

Training

-

Utilization

- **In-Context Learning**: We support various ICL strategies, including `KATE`, `GlobalE`, and `APE`.
- **Chain-of-Thought**: For some datasets, we support three types of CoT evaluation: `base`, `least-to-most`, and `pal`.
- **Ranking Types**: We currently support three ranking types for MultipleChoiceDataset.
- **Prefix Caching**: By caching the `past_key_value` of prefix, we can speed up local inference by up to 6x.
- **vLLM and Flash Attention Support**: We also support [`vLLM`](https://github.com/vllm-project/vllm) and [`Flash Attention`](https://github.com/Dao-AILab/flash-attention) for efficient inference.
- **Quantization**: BitsAndBytes and GPTQ quantization are supported.


## Quick Start

### Install

```python
git clone https://github.com/RUCAIBox/LLMBox.git && cd LLMBox
pip install -r requirements.txt
```

### Quick Start with Training

To pre-train a LLaMA-2 7B model with default settings, you can run the following command:

```bash
cd training
bash bash/run_7b_pt.sh
```

For further supervised fine-tuning, you can run the following command to fine-tune with deepspeed3:

```bash
bash bash/run_7b_ds3.sh
```


### Quick Start with Utilization

To utilize your model or compare with other models, you can run the following command. This is default to run the OpenAI GPT 3.5 turbo model on the CoPA dataset in a zero-shot manner.

```python
python inference.py -m gpt-3.5-turbo -d copa  # --num_shot 0 --model_type instruction
```


## Training



```python
python train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/ \
    --dataset alpaca_data_1k.json \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --save_strategy "epoch" \
    --save_steps 2 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "constant"
```

For more details, view the [training](./training/README.md) documentation.

## Utilization

We provide a broad support on Huggingface models, OpenAI and Anthropic models for further utilization. Currently a total of 51 commonly used datasets are supported. For a full list of supported models and datasets, view the [llmbox](./llmbox/README.md) documentation.

```python
```

We enable efficient evaluation methods by default. Because of the reproducibility problems of vllm, you can use the following command to toggle vllm and prefix caching:

```python
python inference.py -m model_path_or_name -d dataset_name --vllm False --prefix_caching True --flash_attention True
```

<table>
    <tr>
        <td colspan=4 align="center"><b>Utilization</b></td>
    </tr>
    <tr>
        <td rowspan=2><b>Dataset</b></td>
        <td><code>get_ppl</code></td>
        <td><code>get_prob</code></td>
        <td><code>generation</code></td>
    </tr>
    <tr>
        <td><b>Hellaswag (0-shot)</b></td>
        <td><b>MMLU (5-shot)</b></td>
        <td><b>GSM (8-shot)</b></td>
    </tr>
    <tr>
        <td><b>GPT-4</b></td>
        <td>76.01</td>
        <td>45.97</td>
        <td>14.56</td>
    </tr>
    <tr>
        <td><b>LLaMA-2 (70B)</b></td>
        <td>76</td>
        <td>45.95</td>
        <td>14.63</td>
    </tr>
</table>

<!-- For a full list of evaluation results, view our paper. -->

## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/LLMBox/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

Make sure to format your code with `yapf --style style.cfg` and `isort` before submitting a PR.


## The Team

LLMBox is developed and maintained by [AI Box](http://aibox.ruc.edu.cn/).

## License

LLMBox uses [MIT License](./LICENSE).

## Reference

If you find LLMBox useful for your research or development, please cite the following papers:

```
```
