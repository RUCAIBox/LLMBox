# How to Use Chat Template

## What is Chat Template

If you are using an instruct-tuned large language models (LLMs), you need a chat template to correctly prompt the model. Different models are trained with different input formats. This is especially noted in `transformers`'s new `chat_template` feature.

Newer models like LLaMA-3 include the chate template in the `tokenizer_config.json` file. However, other popular models like Vicuna do not have this feature built-in. LLMBox provides a simple way to use chat templates with any model.

For more details, please refer to [this repo](https://github.com/chujiezheng/chat_templates) and [huggingface's documentation](https://huggingface.co/docs/transformers/chat_templating).

## How to Use Chat Template in LLMBox

### Load automatically

In most cases, [LLMBox](https://github.com/RUCAIBox/LLMBox) detects the model type (whether it is a pre-trained model or a chat model). If it is a chat model, we will automatically use the chat template feature.

> [!TIP]
> You can also manually set the model type with `--model_type base` or `--model_type chat`.

Currently we support 7 chat templates including `base` (default), `llama3`, `chatml`, `llama2`, `zephyr`, `phi3`, and `alpaca` (find in [here](https://github.com/RUCAIBox/LLMBox/blob/main/utilization/chat_templates.py)). For example, if you are using the `llama3` model, we will automatically load the `llama3` chat template, which looks like this:

```python
"llama3": {
    "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
    "system_end": "<|eot_id|>",
    "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
    "user_end": "<|eot_id|>",
    "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "assistant_end": "<|eot_id|>",
    "auto_leading_space": True,
    "default_stops": ["<|eot_id|>"],
}
```

When loading a chat-based model, we try to match the model with the chat template by the model's name. For example, the `Meta-Llama3-8B-Instruct` model will be matched with the `llama3` chat template.

### Specify a supported chat template

If the chat template is not correctly loaded, you can manually set the chat template by adding the `--chat_template` argument to the command line.

For example, InternLM-2 uses the `chatml` chat template. You can specify the chat template like this:

```bash
python inference.py -m internlm/internlm2-chat-7b -d gsm8k --chat_template chatml
```

### Use the chat template that comes with the tokenizer

In the above examples, we use our own chat templates, which are needed for some evaluation setups (e.g., `ppl_no_option`). It is because a more fine-grained control is needed for those setups.

However, you can still use the chat template that comes with the tokenizer. For example, if you are using the `Meta-Llama3-8B-Instruct` model, you can use the chat

```bash
python inference.py -m Meta-Llama3-8B-Instruct -d gsm8k --chat_template tokenizer
```

Alternatively, you can use the `--chat_template` argument to specify the path to a jinja template file.

For example, if you have a custom chat template `custom_chat_template.jinja`, you can load it like this:

```bash
python inference.py -m Meta-Llama3-8B-Instruct -d gsm8k --chat_template path/to/custom_chat_template.jinja
```
