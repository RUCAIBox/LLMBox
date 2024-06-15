# [Trouble Shooting] How to Debug an Evaluation Run

This tutorial focuses on debugging an evaluation run that does not reproduce the expected results. This can be caused by the model not generating any predictions, the dataset not being formatted correctly, or the metrics not being calculated correctly. This tutorial will guide you through the process of locating the problem and fixing it.

## Locating the Problem

Every run of the model should produce an evaluation results file, which contains the input data and the model's predictions. You can find this file in the `evaluation_results` folder.

If the process ended normally, the file should be a valid JSON file with metrics:

```json
[
    {
        "index":0,
        "source":"<|start_header_id|>user<|end_header_id|>\n\nAnswer the following question.\n\nQuestion: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer: <|eot_id|> <|start_header_id|>assistant<|end_header_id|>",
        "raw_prediction":[
            "\n\nLet's break this down step by step!\n\nJanet's ducks lay 16 eggs per day. She eats 3 for breakfast, so that leaves:\n\n16 - 3 = 13 eggs\n\nShe bakes muffins with 4 eggs, so that leaves:\n\n13 - 4 = 9 eggs\n\nShe sells the remaining 9 eggs at the farmers' market. Each egg sells for $2, so she makes:\n\n9 eggs x $2 per egg = $18\n\nJanet makes $18 every day at the farmers' market."
        ],
        "processed_prediction":[
            "18"
        ],
        "reference":"18",
        "metric":{
            "Accuracy":true
        },
        "subset":null
    },
    ...
]
```

Alternatively, if the process was ended prematurely, the file will be a valid jsonlines file:

```json
{"index": 0, "source": ["(\"<|start_header_id|>user<|end_header_id|>\\n\\nAnswer the following question.\\n\\nQuestion: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\\nAnswer: <|eot_id|> <|start_header_id|>assistant<|end_header_id|>\",)"], "raw_prediction": "\n\nLet's break this down step by step!\n\nJanet's ducks lay 16 eggs per day.\nShe eats 3 eggs for breakfast, so that leaves 16 - 3 = 13 eggs.\nShe bakes muffins with 4 eggs, so that leaves 13 - 4 = 9 eggs.\nShe sells the remaining 9 eggs at the farmers' market.\n\nEach egg sells for $2, so she makes:\n9 eggs x $2 per egg = $18\n\nJanet makes $18 every day at the farmers' market.", "reference": "18"}
...
```

You can look into the evaluation reulsts file to see if the model is generating normally.

1. If the `raw_prediction` field is empty, the model is not generating any predictions. This might because of the model encountering a stop sequence in output. You can check the `stop` field in the generation arguments and `default_stops` in the chat_templates configuration.

2. If the `raw_prediction` field seems to be normal, you can check the `processed_prediction` field to see if the answer is being extracted correctly in the `post_processing` step.

3. If the `raw_prediction` field continues to output after the completion of the output, it may be that the stop sequence has not been correctly configured. You can check the `stop` field in the generation arguments and the chat_templates configuration.

4. If the `reference` field is not formatted as expected, it may be that the dataset is not formatted correctly. You can check the `references` property in the dataset class is correctly formatted.

5. If everything seems to be normal, you can check the `metric` to see if the metrics are being calculated correctly, especially if the metric is complex.

## Fixing the Problem

If you have located the problem, you can try to fix it by following the steps below.

### Checking the `stop` Generation Argument

The `stop` argument is a list of strings that the model will stop generating after encountering. You can check the `stop` field in the log to see if the model is correctly configured.

**HuggingFace Models:**

```text
2024-06-15 19:30:19 INFO batch_sampler.py:38 Evaluating generation on mt_bench (model_attr={'model_type': 'chat', 'model_backend': 'huggingface', 'model_max_input': 8192, 'model_max_input_and_output': 8192, 'multi_turn': True}, generation_kwargs={'max_new_tokens': 1024, 'stopping_criteria': [KeyWordsCriteria(stop_sequences=[[128009]])], 'pad_token_id': 128000, 'eos_token_id': 128001}, num_shots=0, len=1, num_instances=1, use_cache=False)
```

We convert the `stop` field to a list of integers in the `stopping_criteria` field. In the above example, the stop sequence is `[128009]`, which corresponds to the `<|eot_id|>` token.

**vLLM Models:**

```text
2024-06-15 20:10:33 INFO batch_sampler.py:38 Evaluating generation on mt_bench (model_attr={'model_type': 'chat', 'model_backend': 'vllm'}, generation_kwargs=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['<|eot_id|>'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=1024, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), num_shots=0, len=80, num_instances=80, use_cache=False)
```

LLaMA-3's default stop are `['<|eot_id|>']`.


**API Models:**

The following model does not use `stop`:

```text
2024-06-15 20:35:50 INFO batch_sampler.py:38 Evaluating generation on mt_bench (model_attr={'model_type': 'chat', 'model_backend': 'openai', 'multi_turn': True}, generation_kwargs={'max_tokens': 4096, 'seed': 2023}, num_shots=0, len=1, num_instances=1, use_cache=False)
```

While the following one uses `stop`:

```text
2024-06-15 20:39:37 INFO batch_sampler.py:38 Evaluating generation on drop (model_attr={'model_type': 'chat', 'model_backend': 'openai'}, generation_kwargs={'max_tokens': 64, 'seed': 2023, 'stop': ['\n'], 'temperature': 0}, num_shots=0, len=1, num_instances=1, use_cache=False)
```

**`stop` might be set in the following places:**

1. In the `init_arguments` method or the class variable of the dataset class

2. In the command line arguments `stop`

3. In the chat template `default_stop`

4. In the `transform` validation of generation arguments (Anthropic models does not support a whitespace stop)

### Checking the Chat Template Configuration

If you are using an instruct-tuned model, you need a chat template to correctly prompt the model. Different models may require different chat templates.

Currently we support 7 chat templates including `base` (default), `llama3`, `chatml`, `llama2`, `zephyr`, `phi3`, and `alpaca`. This offers a more fine-grained control over the chat format.

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

When loading a chat-based model, i.e. setting `--model_type chat`, we try to match the model with the chat template by the model's name. For example, the `Meta-Llama3-8B-Instruct` model will be matched with the `llama3` chat template.

You can check that the chat template is correctly loaded in the log:

```text
2024-06-15 20:39:37 INFO Automatically set chat_template to llama3.
```

If the chat template is not correctly loaded, you can manually set the chat template by adding the `--chat_template` argument to the command line.

```bash
python inference.py -m internlm/internlm2-chat-7b -d gsm8k --chat_template chatml
```

If the chat format is not supported by LLMBox, you can create a new chat template by extending the [`chat_templates.py`](https://github.com/RUCAIBox/LLMBox/tree/main/utilization/chat_templates.py) file.

Alternatively, you can also pass in a jinja2 template string, which is also compatible with HuggingFace's `tokenizers` library.

### Checking the `default_stops` in the Chat Template

In rare cases, you may want to modify the `default_stops` field in the chat template configuration.

If the `default_stops` field prevents the model from generating output, you can try overwriting the `default_stops` arguments with an empty string.

```bash
python inference.py -m Meta-Llama3-8B-Instruct -d gsm8k --default_stops ""
```

If you need to extend the `default_stops` field in the chat template configuration.

```bash
python inference.py -m Meta-Llama3-8B-Instruct -d gsm8k --default_stops "<|eot_id|>" "<|start_header_id|>"
```

### Checking the `post_processing` Step

The `post_processing` step is used to extract the answer from the model's output. If the `post_processing` step is not correctly configured, the model will not be able to extract the answer correctly.

You can first locate the dataset class in the [`utilization/dataset`](https://github.com/RUCAIBox/LLMBox/tree/main/utilization/dataset) folder and check the `post_processing` method.

```python
class Drop(GenerationDataset):

    ...

    @staticmethod
    def post_processing(predictions):
        new_predictions = []
        pattern = r"[.!(\n)]"
        for pred in predictions:
            match = re.search(pattern, pred)
            if match:
                index = match.start()
                pred = pred[:index]
            new_predictions.append(pred)
        return new_predictions
```

### Checking the `references` Property

The `references` property in the dataset class is used to check the model output against the reference answer. If the `references` property is not formatted correctly, the model will not be able to calculate the metrics correctly.


```python
class Drop(GenerationDataset):

    ...

    @cached_property
    def references(self):
        return [instance["answers_spans"]["spans"] for instance in self.evaluation_data]
```

### Checking the Metric Calculation

```python
class Drop(GenerationDataset):

    metrics = [F1(force_number_match=True, word_tokenize="regex", align_bag="counter"), Em()]

    ...
```

If you found `processed_prediction` matches the `reference` field, but the metric is still not calculated correctly, you can check the metric calculation method in the dataset class.


```python
class F1(Metric):

    def __init__(
        self,
        *,
        dataset: Literal["independent"] = "independent",
        multiref_strategy: Literal["max", "leave_one_out"] = "max",
        word_tokenize: Literal["nltk", "split", "regex"] = "nltk",
        normalize_level: Literal["token", "text", "both"] = "both",
        align_bag: Literal["counter", "set"] = "counter",
        force_number_match=False,
    ):
        self.dataset = dataset
        self.word_tokenize = _TOKENIZER_DICT[word_tokenize]
        self.normalize_level = normalize_level
        self.multiref_strategy = multiref_strategy
        self.align_bag = align_bag
        self.force_number_match = force_number_match
    ...

```

## In Closing

If you still have any problems replicating an evaluation run, please feel free to reach out to us by [creating an issue](https://github.com/RUCAIBox/LLMBox/issue).

You can attach the log file and evaluation results file to the issue, and we will help you locate the problem.
