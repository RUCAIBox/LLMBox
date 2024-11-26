[LLMBox](https://github.com/RUCAIBox/LLMBox) | [Training](https://github.com/RUCAIBox/LLMBox/tree/main/training) | **Utilization**

# Utilization

- [Utilization](#utilization)
  - [Supported Datasets](#supported-datasets)
  - [Customize Dataset](#customize-dataset)
  - [Usage](#usage)
    - [Model Arguments](#model-arguments)
    - [Dataset Arguments](#dataset-arguments)
    - [Evaluation Arguments](#evaluation-arguments)
  - [Supported Models](#supported-models)
  - [Customize Model](#customize-model)
    - [Customizing HuggingFace Models](#customizing-huggingface-models)
    - [Adding a New Model Provider](#adding-a-new-model-provider)
  - [Customize Chat Template](#customize-chat-template)
  - [Change Log](#change-log)


## Supported Datasets

- See a full list of supported datasets at [here](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/supported-datasets.md).
- See how to [load datasets with subsets](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/how-to-load-datasets-with-subsets.md).
- See how to [load datasets](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/how-to-load-datasets-from-huggingface.md) from Hugging Face or its mirror.

## Customize Dataset

See [this guide](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/how-to-customize-dataset.md) for details.


## Usage

Evaluating davinci-002 on HellaSwag, with prefix caching and flash attention enabled by default:

```bash
python inference.py -m davinci-002 -d hellaswag
```

Evaluating Gemma on MMLU:

```bash
python inference.py -m gemma-7b -d mmlu -shots 5
```

This will report the 57 subsets of MMLU, along with the macro average performance on four categories.

Evaluating Phi-2 on GSM8k using self-consistency and 4-bit quantization:

```bash
python inference.py -m microsoft/phi-2 -d gsm8k -shots 8 --sample_num 100 --load_in_4bit
```

Evaluating LLaMA-2 (7b) on CMMLU and CEval with instruction using vllm:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py -m llama-2-7b-hf -d cmmlu ceval --vllm True --model_type chat
```

We use all cuda devices by default. You can specify the device with `CUDA_VISIBLE_DEVICES`.

### Model Arguments

Define the model parameters, efficient evaluation settings, generation arguments, quantization, and additional configuration options.

We provide an enumeration ([`model_enum`](https://github.com/RUCAIBox/LLMBox/tree/main/utilization/model_enum.py)) for models corresponding to each `model_backend`. If a model is not listed within this enumeration, `--model_backend` should be specified directly.


```text
--model_name_or_path MODEL_NAME_OR_PATH, --model MODEL_NAME_OR_PATH, -m MODEL_NAME_OR_PATH
                      The model name or path, e.g., davinci-002, meta-
                      llama/Llama-2-7b-hf, ./mymodel (default: None)
--model_type {base,instruction}
                      The type of the model, which can be chosen from `base`
                      or `instruction`. (default: base)
--model_backend {anthropic,dashscope,huggingface,openai,qianfan,vllm}
                      The model backend
--device_map DEVICE_MAP
                      The device map for model and data (default: auto)
--vllm [VLLM]         Whether to use vllm (default: False)
--flash_attention [FLASH_ATTENTION]
                      Whether to use flash attention (default: True)
--no_flash_attention  Whether to use flash attention (default: False)
--openai_api_key OPENAI_API_KEY
                      The OpenAI API key (default: None)
--anthropic_api_key ANTHROPIC_API_KEY
                      The Anthropic API key (default: None)
--dashscope_api_key DASHSCOPE_API_KEY
                      The Dashscope API key (default: None)
--qianfan_access_key QIANFAN_ACCESS_KEY
                      The Qianfan access key (default: None)
--qianfan_secret_key QIANFAN_SECRET_KEY
                      The Qianfan secret key (default: None)
--tokenizer_name_or_path TOKENIZER_NAME_OR_PATH, --tokenizer TOKENIZER_NAME_OR_PATH
                      The tokenizer name or path, e.g., cl100k_base, meta-llama/Llama-2-7b-hf, ./mymodel
```

Generation arguments and quantization options:

```txt
--max_tokens MAX_TOKENS
                      The maximum number of tokens for output generation
                      (default: None)
--max_length MAX_LENGTH
                      The maximum number of tokens of model input sequence
                      (default: None)
--temperature TEMPERATURE
                      The temperature for models (default: None)
--top_p TOP_P         The model considers the results of the tokens with
                      top_p probability mass. (default: None)
--top_k TOP_K         The model considers the token with top_k probability.
                      (default: None)
--frequency_penalty FREQUENCY_PENALTY
                      Positive values penalize new tokens based on their
                      existing frequency in the generated text, vice versa.
                      (default: None)
--repetition_penalty REPETITION_PENALTY
                      Values>1 penalize new tokens based on their existing
                      frequency in the prompt and generated text, vice
                      versa. (default: None)
--presence_penalty PRESENCE_PENALTY
                      Positive values penalize new tokens based on whether
                      they appear in the generated text, vice versa.
                      (default: None)
--stop STOP [STOP ...]
                      List of strings that stop the generation when they are
                      generated. E.g. --stop 'stop' 'sequence' (default:
                      None)
--no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE
                      All ngrams of that size can only occur once. (default:
                      None)
--best_of BEST_OF, --num_beams BEST_OF
                      The beam size for beam search (default: None)
--length_penalty LENGTH_PENALTY
                      Positive values encourage longer sequences, vice
                      versa. Used in beam search. (default: None)
--early_stopping [EARLY_STOPPING]
                      Positive values encourage longer sequences, vice
                      versa. Used in beam search. (default: None)
--system_prompt SYSTEM_PROMPT, -sys SYSTEM_PROMPT
                      The system prompt for chat-based models
--chat_template CHAT_TEMPLATE
                      The chat template for local chat-based models. Support model default chate template (choose from 'base', 'llama2', 'chatml', 'zephyr', 'phi3', 'llama3', ...) or standard HuggingFace tokenizers chat template
--bnb_config BNB_CONFIG
                      JSON string for BitsAndBytesConfig parameters.
--load_in_8bit [LOAD_IN_8BIT]
                      Whether to use bnb's 8-bit quantization to load the
                      model. (default: False)
--load_in_4bit [LOAD_IN_4BIT]
                      Whether to use bnb's 4-bit quantization to load the
                      model. (default: False)
--gptq [GPTQ]         Whether the model is a gptq quantized model. (default:
                      False)
--vllm_gpu_memory_utilization VLLM_GPU_MEMORY_UTILIZATION
                      The maximum gpu memory utilization of vllm. (default:
                      None)
--torch_dtype {float16,bfloat16,float32}
                      The torch dtype for model input and output
```

### Dataset Arguments

Configure dataset parameters such as the dataset identifiers, batch size, example strategies, chain-of-thought (CoT) strategies, and other relevant settings.

You can evaluate datasets sequentially in a single run when they require similar evaluation parameters. Both `evaluation_set` and `example_set` support the Huggingface [String API](https://huggingface.co/docs/datasets/loading#slice-splits) for defining dataset slices.

```text
--dataset_names DATASET [DATASET ...], -d DATASET [DATASET ...], --dataset DATASET [DATASET ...]
                      Space splitted dataset names. If only one dataset is specified, it can be followed by
                      subset names or category names. Format: 'dataset1 dataset2', 'dataset:subset1,subset2', or
                      'dataset:[cat1],[cat2]', e.g., 'copa race', 'race:high', 'wmt16:en-ro,en-fr', or
                      'mmlu:[stem],[humanities]'. (default: None)
--batch_size BATCH_SIZE, -bsz BATCH_SIZE, -b BATCH_SIZE
                      The evaluation batch size. Specify an integer (e.g., '10') to use a fixed batch size for
                      all iterations. Alternatively, append ':auto' (e.g., '10:auto') to start with the specified
                      batch size and automatically adjust it in subsequent iterations to maintain constant CUDA
                      memory usage (default: 1)
--dataset_path DATASET_PATH
                      The path of dataset if loading from local. Supports
                      repository cloned from huggingface, dataset saved by
                      `save_to_disk`, or a template string e.g.
                      'mmlu/{split}/{subset}_{split}.csv'. (default: None)
--evaluation_set EVALUATION_SET
                      The set name for evaluation, supporting slice, e.g.,
                      validation, test, validation[:10] (default: None)
--example_set EXAMPLE_SET
                      The set name for demonstration, supporting slice,
                      e.g., train, dev, train[:10] (default: None)
--instruction INSTRUCTION
                      The format to format the instruction for each instance. Either f-string or jinja2 format is supported. E.g., 'Answer the following question: {question}\nAnswer:'"
--num_shots NUM_SHOTS, -shots NUM_SHOTS
                      The few-shot number for demonstration (default: 0)
--max_example_tokens MAX_EXAMPLE_TOKENS
                      The maximum token number of demonstration (default:
                      1024)
```

Different types of datasets support different evaluation methods. The following table lists the supported evaluation methods and prompting methods for each dataset type.

<table>
    <tr>
        <td><b>Dataset</b></td>
        <td><b>Evaluation Method</b></td>
        <td><b>Prompt</b></td>
    </tr>
    <tr>
        <td><p><b>Generation</b></p>
        <p><pre><code>{
  "question":
    "when was ...",
  "answer": [
    '14 December 1972',
    'December 1972'
  ]
}</code></pre></p></td>
        <td><p><code>generation</code></p><p>Generate based on the source text</p></td>
        <td><p><pre><code>Q: When was ...?
A: ________</code></pre></p></td>
    </tr>
    <tr>
        <td rowspan=3><p><b>MultipleChoice</b></p>
<pre><code>{
  "question":
    "What is the ...?",
  "choices": [
    "The first",
    "The second",
    ...
  ],
  "answer": 3
}</code></pre></td>
        <td rowspan=2><p><code>get_ppl</code></p><p>Calculate perplexity of the option text based on the source text</p></td>
        <td><p style="text-align: center;"><code>ppl_no_option</code></p>
<p><pre><code>Q: What is ...?
A: The first
   └--ppl--┘</code></pre></p></td>
    </tr>
    <tr>
        <td><p style="text-align: center;"><code>ppl</code></p>
<p><pre><code style="border-style: solid;">Q: What is ...?
A. The first
B. The second
C. ...
A: A. The first
   └----ppl---┘</code></pre></p></td>
    </tr>
    <tr>
        <td><p><code>get_prob</code></p><p>Get the probability of each option label</p></td>
        <td><p style="text-align: center;"><code>prob</code></p>
<p><pre><code>Q: What is ...?
A. The first
B. The second
C. ...
A: _
   └→ [A B C D]</code></pre></p></td>
    </tr>
</table>

GetPPL:

![get_ppl](https://github.com/user-attachments/assets/5b420450-223e-4bb6-a92e-feb7c0ddc5b0)

GetProb:

![get_prob](https://github.com/user-attachments/assets/53b3253b-9c08-4a06-bc2d-43c03ac68611)


```text
--ranking_type {ppl,prob,ppl_no_option}
                      The evaluation and prompting method for ranking task
                      (default: ppl_no_option)
--sample_num SAMPLE_NUM, --majority SAMPLE_NUM, --consistency SAMPLE_NUM
                      The sampling number for self-consistency (default: 1)
--kate [KATE], -kate [KATE]
                      Whether to use KATE as an ICL strategy (default:
                      False)
--globale [GLOBALE], -globale [GLOBALE]
                      Whether to use GlobalE as an ICL strategy (default:
                      False)
--ape [APE], -ape [APE]
                      Whether to use APE as an ICL strategy (default: False)
--cot {base,least_to_most,pal}
                      The method to prompt, eg. 'base', 'least_to_most',
                      'pal'. Only available for some specific datasets.
                      (default: None)
--perspective_api_key PERSPECTIVE_API_KEY
                      The Perspective API key for toxicity metrics (default:
                      None)
--pass_at_k PASS_AT_K
                      The k value for pass@k metric (default: None)
```

### Evaluation Arguments

Specify the random seed, logging directory, evaluation results directory, and other arguments.

```text
--seed SEED           The random seed (default: 2023)
--logging_dir LOGGING_DIR
                      The logging directory (default: logs)
--log_level {debug,info,warning,error,critical}
                      Logger level to use on the main node. Possible choices
                      are the log levels as strings: 'debug', 'info',
                      'warning', 'error' and 'critical' (default: info)
--evaluation_results_dir EVALUATION_RESULTS_DIR
                      The directory to save evaluation results, which
                      includes source and target texts, generated texts, and
                      the references. (default: evaluation_results)
--log_results [LOG_RESULTS]
                      Whether to log the evaluation results. Notes that the generated JSON file will be the same
                      size as the evaluation dataset itself
--no_log_results      Whether to log the evaluation results. Notes that the generated JSON file will be the same
                      size as the evaluation dataset itself
--dry_run [DRY_RUN]   Test the evaluation pipeline without actually calling
                      the model. (default: False)
--proxy_port PROXY_PORT
                      The port of the proxy (default: None)
--dataset_threading [DATASET_THREADING]
                      Load dataset with threading
--no_dataset_threading
                      Load dataset with threading
--dataloader_workers DATALOADER_WORKERS
                      The number of workers for dataloader
```

## Supported Models

🔥 New models supported: `Llama3` series, `Gemma2` series.

<table>
  <tr>
      <td><b>Backend</b></td>
      <td><b>Entrypoint</b></td>
      <td><b>Example Model</b></td>
      <td><b>Supported Methods</b></td>
  </tr>
  <tr>
      <td>Huggingface</td>
      <td>AutoModelForCasualLM</td>
      <td><code>Llama-2-7b-hf</code>, <code>Meta-Llama3-8B-Instruct</code></td>
      <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
  </tr>
  <tr>
      <td rowspan=3>OpenAI<br><code>openai>=1.0.0</code></td>
      <td>Chat Completion Models</td>
      <td><code>gpt-4o</code>, <code>gpt-4-0125-preview</code></td>
      <td><code>generation</code>
  </tr>
  <tr>
      <td>Completion Models (Legacy)</td>
      <td><code>davinci-002</code></td>
      <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
  </tr>
  <tr>
      <td>OpenAI-compatible APIs*</td>
      <td><code>llama-3-sonar-small-32k-chat</code>, <code>deepseek-chat</code></td>
      <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
  </tr>
  <tr>
      <td>Qianfan</td>
      <td>Chat Completion Models</td>
      <td><code>ernie-speed-8k</code></td>
      <td><code>generation</code></td>
  </tr>
  <tr>
      <td>Dashscope</td>
      <td>Generation</td>
      <td><code>qwen-turbo</code></td>
      <td><code>generation</code></td>
  </tr>
  <tr>
      <td>Anthropic</td>
      <td>Chat Completion Models</td>
      <td><code>claude-3-haiku-20240307</code></td>
      <td><code>generation</code></td>
  </tr>
  <tr>
      <td>vLLM<br><code>vllm>=0.4.3</code></td>
      <td>LLM</td>
      <td><code>Llama-2-7b-hf</code>, <code>Meta-Llama3-8B-Instruct</code></td>
      <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
  </tr>
</table>

For openai-compatible models like [Perplexity](https://docs.perplexity.ai/docs/getting-started), you can use the `--model_backend openai` argument to use openai python library and `OPENAI_BASE_URL` to specify the base URL.

```bash
OPENAI_BASE_URL=https://api.perplexity.ai python inference.py -m llama-3-sonar-small-32k-chat -d hellaswag --openai_api_key PERPLEXITY_API_KEY --model_backend openai
```

In some cases (e.g. evaluating with `get_prob`), you may need to specify the `--tokenizer` to load the correct tokenizer (e.g. `cl100k_base`).

> [!TIP]
> Use dotenv `.env` file to store your API keys and other sensitive information. The `.env` file should be in the root directory of the project.


## Customize Model

### Customizing HuggingFace Models

If you are building on your own model, such as using a fine-tuned model, you can evaluate it easily from python script. Detailed steps and example code are provided in the [customize HuggingFace model guide](https://github.com/RUCAIBox/LLMBox/tree/main/docs/examples/customize_huggingface_model.py).

### Adding a New Model Provider

If you're integrating a new model provider, begin by extending the [`Model`](https://github.com/RUCAIBox/LLMBox/tree/main/utilization/model/model.py) class. Implement essential methods such as `generation`, `get_ppl` (get perplexity), and `get_prob` (get probability) to support different functionalities. For instance, here's how you might implement the `generation` method for a new model:

```python
class NewModel(Model):

    model_backend = "new_model"

    def call_model(self, batched_inputs: List[str]) -> List[Any]:
        return ...  # call to model, e.g., self.model.generate(...)

    def to_text(self, result: Any) -> str:
        return ...  # convert result to text, e.g., result['text']

    def generation(self, batched_inputs: List[str]) -> List[str]:
        results = self.call_model(batched_inputs)
        results = [to_text(result) for result in results]
        return results
```

And then, you should register your model in the [`load`](https://github.com/RUCAIBox/LLMBox/tree/main/utilization/model/load.py) file.

## Customize Chat Template

Chat templates are used to formatting conversational messages to text input for local chat-based models.

```bash
python inference.py -m Meta-Llama-3-8B-Instruct -d gsm8k --model_type chat --chat_template llama3 -shots 8 -sys "You are a helpful assistant."
```

You don't need to specify the chat template for hosted commercial APIs.

```bash
python inference.py -m gpt-3.5-turbo -d gsm8k --model_type chat -shots 8 -sys "You are a helpful assistant."
```

For more details, view [how to use chat template](https://github.com/RUCAIBox/LLMBox/blob/main/docs/utilization/how-to-use-chat-template.md).


## Change Log

- **June 6, 2024**: Refactor the codebase and add support for hf-mirror.
- **May 24, 2024**: Chat format support including conversational few-shot and system prompts.
- **May 10, 2024**: New instruction formatting using f-string and jinja2.
- **May 7, 2024**: Bump openai and vllm version.
- **Apr 16, 2024**: Full support for KV caching.
- **March 18, 2024**: First release of LLMBox.
