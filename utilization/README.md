[LLMBox](..) | [Training](../training) | **Utilization**

# Utilization

- [Utilization](#utilization)
  - [Usage](#usage)
    - [Model Arguments](#model-arguments)
    - [Dataset Arguments](#dataset-arguments)
    - [Evaluation Arguments](#evaluation-arguments)
  - [Supported Models](#supported-models)
  - [Customize Model](#customize-model)
  - [Customize Chat Template](#customize-chat-template)
  - [Supported Datasets](#supported-datasets)
  - [Customize Dataset](#customize-dataset)
  - [Change Log](#change-log)

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
CUDA_VISIBLE_DEVICES=0 python inference.py -m llama-2-7b-hf -d cmmlu ceval --vllm True --model_type instruction
```

We use all cuda devices by default. You can specify the device with `CUDA_VISIBLE_DEVICES`.

### Model Arguments

Define the model parameters, efficient evaluation settings, generation arguments, quantization, and additional configuration options.

We provide an enumeration ([`enum`](model/enum.py)) for models corresponding to each `model_backend`. If a model is not listed within this enumeration, `--model_backend` should be specified directly.


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
      <td><code>Llama-2-7b-hf</code></td>
      <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
  </tr>
  <tr>
      <td rowspan=2>OpenAI</td>
      <td>Chat Completion Models</td>
      <td><code>gpt-4-0125-preview</code>, <code>deepseek-chat</code></td>
      <td><code>generation</code>
  </tr>
  <tr>
      <td>Completion Models (Legacy)</td>
      <td><code>davinci-002</code></td>
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
      <td>vLLM</td>
      <td>LLM</td>
      <td><code>Llama-2-7b-hf</code></td>
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

By inheriting the [`Model`](model/model.py) class, you can customize support for more models. You can implement `generation`, `get_ppl`, and `get_prob` methods to support different models. For example, you can implement the `generation` method for a new model as follows:

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

And then, you should register your model in the [`load`](model/load.py) file.

## Customize Chat Template

Chat templates are used to formatting conversational messages to text input for local chat-based models.

```bash
python inference.py -m Meta-Llama-3-8B-Instruct -d gsm8k --model_type chat --chat_template llama3 -shots 8 -sys "You are a helpful assistant."
```

You don't need to specify the chat template for hosted models.

```bash
python inference.py -m gpt-3.5-turbo -d gsm8k --model_type chat -shots 8 -sys "You are a helpful assistant."
```

You can customize the [chat template](https://github.com/RUCAIBox/LLMBox/blob/main/utilization/chat_templates.py) for local chat-based models. We provide a set of chat templates for different models. You can specify a jinja2 chat template with the `--chat_template` argument. It works in the same way as the [tokenizers](https://huggingface.co/docs/transformers/main/en/chat_templating).


## Supported Datasets

We currently support 53 commonly used datasets for LLMs. Each dataset may includes multiple subsets, or is a subset of a collection.

Load from huggingface server:
```bash
python inference.py -d copa
python inference.py -d race:middle,high
python inference.py -d race:middle,high --evaluation_set "test[:10]" --example_set "train"
```

<table>
  <tr>
      <td><b>Dataset</b></td>
      <td><b>Subsets / Collections</b></td>
      <td><b>Evaluation Type</b></td>
      <td><b>CoT</b></td>
      <td><b>Notes</b></td>
  </tr>
  <tr>
      <td rowspan=3>AGIEval(<code>agieval</code>, alias of <code>agieval_single_choice</code> and <code>agieval_cot</code>)</td>
      <td><b>English</b>: <code>sat-en</code>, <code>sat-math</code>, <code>lsat-ar</code>, <code>lsat-lr</code>, <code>lsat-rc</code>, <code>logiqa-en</code>, <code>aqua-rat</code>, <code>sat-en-without-passage</code></td>
      <td rowspan=2>MultipleChoice</td>
      <td></td>
      <td rowspan=3></td>
  </tr>
  <tr>
      <td><code>gaokao-chinese</code>, <code>gaokao-geography</code>, <code>gaokao-history</code>, <code>gaokao-biology</code>, <code>gaokao-chemistry</code>, <code>gaokao-english</code>, <code>logiqa-zh</code></td>
      <td></td>
  </tr>
  <tr>
      <td><code>jec-qa-kd</code>, <code>jec-qa-ca</code>, <code>math</code>, <code>gaokao-physics</code>, <code>gaokao-mathcloze</code>, <code>gaokao-mathqa</code></td>
      <td>Generation</td>
      <td>✅</td>
  </tr>
  <tr>
      <td>Alpacal Eval (<code>alpaca_eval</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Single GPTEval</td>
  </tr>
  <tr>
      <td>Adversarial Natural Language Inference (<code>anli</code>)</td>
      <td><code>Round2</code> (default)</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>AI2's Reasoning Challenge (<code>arc</code>)</td>
      <td><code>ARC-Easy</code>, <code>ARC-Challenge</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>BIG-Bench Hard (<code>bbh</code>)</td>
      <td><code>boolean_expressions</code>, ...</td>
      <td>Generation</td>
      <td>✅</td>
      <td></td>
  </tr>
  <tr>
      <td>Boolean Questions (<code>boolq</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>CommitmentBank (<code>cb</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td rowspan=4>C-Eval (<code>ceval</code>)</td>
      <td><b>stem</b>: <code>advanced_mathematics</code>, <code>college_chemistry</code>, ...</td>
      <td rowspan=4>MultipleChoice</td>
      <td rowspan=4></td>
      <td rowspan=4></td>
  </tr>
  <tr>
      <td><b>social science</b>: <code>business_administration</code>, <code>college_economics</code>, ...</td>
  </tr>
  <tr>
      <td><b>humanities</b>: <code>art_studies</code>, <code>chinese_language_and_literature</code>, ...</td>
  </tr>
  <tr>
      <td><b>other</b>: <code>accountant</code>, <code>basic_medicine</code>, ...</td>
  </tr>
  <tr>
      <td rowspan=4>Massive Multitask Language Understanding in Chinese (<code>cmmlu</code>)</td>
      <td><b>stem</b>: <code>anatomy</code>, <code>astronomy</code>, ...</td>
      <td rowspan=4>MultipleChoice</td>
      <td rowspan=4></td>
      <td rowspan=4></td>
  </tr>
  <tr>
      <td><b>social science</b>: <code>ancient_chinese</code>, <code>business_ethics</code>, ...</td>
  </tr>
  <tr>
      <td><b>humanities</b>: <code>arts</code>, <code>chinese_history</code>, ...</td>
  </tr>
  <tr>
      <td><b>other</b>: <code>agronomy</code>, <code>chinese_driving_rule</code>, ...</td>
  </tr>
  <tr>
      <td>CNN Dailymail (<code>cnn_dailymail</code>)</td>
      <td><code>3.0.0</code> (default), ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Reasoning About Colored Objects (<code>color_objects</code>)</td>
      <td><i>bigbench</i> (reasoning_about_colored_objects)</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Commonsense QA (<code>commonsenseqa</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Choice Of Plausible Alternatives (<code>copa</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Conversational Question Answering (<code>coqa</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Download: <a href="https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json">train</a>, <a href="https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json">dev</a></td>
  </tr>
  <tr>
      <td>CrowS-Pairs (<code>crows_pairs</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Discrete Reasoning Over the content of Paragraphs (<code>drop</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td rowspan=3>GAOKAO (<code>gaokao</code>)</td>
      <td><b>Chinese</b>: <code>2010-2022_Chinese_Modern_Lit</code>, <code>2010-2022_Chinese_Lang_and_Usage_MCQs</code></td>
      <td rowspan=3>Generation</td>
      <td rowspan=3></td>
      <td rowspan=3>Metric: Exam scoring</td>
  </tr>
  <tr>
      <td><b>English</b>: <code>2010-2022_English_Reading_Comp</code>, <code>2010-2022_English_Fill_in_Blanks</code>, ...</td>
  </tr>
  <tr>
      <td><code>2010-2022_Math_II_MCQs</code>, <code>2010-2022_Math_I_MCQs</code>, ...</td>
  </tr>
  <tr>
      <td>Google-Proof Q&A (<code>GPQA</code>)</td>
      <td><code>gpqa_main</code> (default), <code>gpqa_extended</code>, ...</td>
      <td>MultipleChoiceDataset</td>
      <td>✅</td>
      <td></td>
  </tr>
  <tr>
      <td>Grade School Math 8K (<code>gsm8k</code>)</td>
      <td><code>main</code> (default), <code>socratic</code></td>
      <td>Generation</td>
      <td>✅</td>
      <td>Code exec</td>
  </tr>
  <tr>
      <td>HaluEval(<code>halueval</code>)</td>
      <td><code>dialogue_samples</code>, <code>qa_samples</code>, <code>summarization_samples</code></td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>HellaSWAG (<code>hellaswag</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>HumanEval (<code>humaneval</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Pass@K</td>
  </tr>
  <tr>
      <td>Instruction-Following Evaluation (<code>ifeval</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>LAnguage Modeling Broadened to Account for Discourse Aspects (<code>lambada</code>)</td>
      <td><code>default</code> (default), <code>de</code>, ... (source: <i>EleutherAI/lambada_openai</i>)</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Mathematics Aptitude Test of Heuristics (<code>math</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td>✅</td>
      <td></td>
  </tr>
  <tr>
      <td>Mostly Basic Python Problems (<code>mbpp</code>)</td>
      <td><code>full</code> (default), <code>sanitized</code></td>
      <td>Generation</td>
      <td></td>
      <td>Pass@K</td>
  </tr>
  <tr>
      <td rowspan=4>Massive Multitask Language Understanding(<code>mmlu</code>)</td>
      <td><b>stem</b>: <code>abstract_algebra</code>, <code>astronomy</code>, ...</td>
      <td rowspan=4>MultipleChoice</td>
      <td rowspan=4></td>
      <td rowspan=4></td>
  </tr>
  <tr>
      <td><b>social_sciences</b>: <code>econometrics</code>, <code>high_school_geography</code>, ...</td>
  </tr>
  <tr>
      <td><b>humanities</b>: <code>formal_logic</code>, <code>high_school_european_history</code>, ...</td>
  </tr>
  <tr>
      <td><b>other</b>: <code>anatomy</code>, <code>business_ethics</code>, ...</td>
  </tr>
  <tr>
      <td>Multi-turn Benchmark (<code>mt_bench</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Multi-turn GPTEval</td>
  </tr>
  <tr>
      <td>Natural Questions(<code>nq</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>OpenBookQA (<code>openbookqa</code>)</td>
      <td><code>main</code> (default), <code>additional</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>Penguins In A Table (<code>penguins_in_a_table</code>)</td>
      <td><i>bigbench</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Physical Interaction: Question Answering (<code>piqa</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Question Answering in Context (<code>quac</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>ReAding Comprehension (<code>race</code>)</td>
      <td><code>high</code>, <code>middle</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>Real Toxicity Prompts (<code>real_toxicity_prompts</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td><a href="https://www.perspectiveapi.com/">Perspective</a> Toxicity</td>
  </tr>
  <tr>
      <td>Recognizing Textual Entailment (<code>rte</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Social Interaction QA (<code>siqa</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Stanford Question Answering Dataset (<code>squad, squad_v2</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Story Cloze Test (<code>story_cloze</code>)</td>
      <td><code>2016</code> (default), <code>2018</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td><a href='http://goo.gl/forms/aQz39sdDrO'>Manually download</a></td>
  </tr>
  <tr>
      <td>TL;DR (<code>tldr</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>TriviaQA (<code>triviaqa</code>)</td>
      <td><code>rc.wikipedia.nocontext</code> (default), <code>rc</code>, <code>rc.nocontext</code>, ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>TruthfulQA (<code>truthfulqa_mc</code>)</td>
      <td><code>multiple_choice</code> (default), <code>generation</code> (not supported)</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Vicuna Bench (<code>vicuna_bench</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>GPTEval</td>
  </tr>
  <tr>
      <td>WebQuestions (<code>webq</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Words in Context (<code>wic</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Winogender Schemas (<code>winogender</code>)</td>
      <td><code>main</code>, <code>gotcha</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Group by gender</td>
  </tr>
  <tr>
      <td>WSC273 (<code>winograd</code>)</td>
      <td><code>wsc273</code> (default), <code>wsc285</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>WinoGrande (<code>winogrande</code>)</td>
      <td><code>winogrande_debiased</code> (default), ...</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Conference on Machine Translation (<code>wmt21, wmt19, ...</code>)</td>
      <td><code>en-ro</code>, <code>ro-en</code>, ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Winograd Schema Challenge (<code>wsc</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Extreme Summarization (<code>xsum</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>

</table>

By default we load all the subsets of a dataset:

```bash
python inference.py -m model -d arc
# equivalent: arc:ARC-Easy,ARC-Challenge
```

Unless a default subset is defined:

```bash
python inference.py -m model -d cnn_dailymail
# equivalent: cnn_dailymail:3.0.0
```

Some datasets like GPQA (Google-Proof Q&A) have to load example set separately:

```bash
# few_shot
python inference.py -m model -d gpqa --ranking_type generation -shots 5 --example_set "../gpqa/prompts"
```

If `dataset_path` is not None, the dataset will be loaded from the given local path:

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

---

Also feel free to override this function if you want to load the dataset in a different way:

```python
from .utils import load_raw_dataset_from_file, get_raw_dataset_loader

class MyDataset(Dataset):
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        self.evaluation_data = get_raw_dataset_loader(...)("test")
        self.example_data = load_raw_dataset_from_file("examples.json")
```

## Customize Dataset

We provide two types of datasets: [`GenerationDataset`](dataset/generation_dataset.py), [`MultipleChoiceDataset`](dataset/multiple_choice_dataset.py). You can also customize support for a new dataset type by inheriting the [`Dataset`](dataset/dataset.py) class. For example, you can implement a new `GenerationDataset` as follows:

```python
def NewDataset(GenerationDataset):

    instruction = "Answer the following question.\n\n{source}"
    metrics = [Accuracy()]
    evaluation_set = "test"
    example_set = "dev"
    load_args = ("huggingface/path", "subset")

    extra_model_args = dict(temperature=0)
    category_subsets = {"Group": ["subset1", "subset2"]}

    def format_instance(self, instance):
        src, tgt = func(instance, self.example_data)
        return dict(source=src, target=tgt)

    def reference(self):
        return [i["answer"] for i in self.eval_data]
```

You can load the raw dataset by the following methods:

- Set a `load_args`: The arguments for [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset).
- Or overwrite `load_raw_dataset` function: Set the `self.evaluation_data` and `self.example_data`.

```python
from .utils import load_raw_dataset_from_file, get_raw_dataset_loader

class MyDataset(Dataset):
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        self.evaluation_data = get_raw_dataset_loader(...)("test")
        self.example_data = load_raw_dataset_from_file("examples.json")
```

Then, format the instance by implementing the `instruction` attribute and `format_instance` method. The returned instance should be a dictionary with keys accessible in the instruction string. Both jinja2 and f-string are supported for the instruction string.

<table class="waffle" cellspacing="0" cellpadding="0">
    <thead>
        <tr>
        <th class="column-headers-background"><code>format_instance</code></th>
        <th class="column-headers-background">Type</th>
        <th class="column-headers-background"><code>instruction</code></th>
        <th class="column-headers-background">Type</th>
        <th class="column-headers-background">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td class="s1" dir="ltr">customized</td>
        <td class="s1" dir="ltr"><code>Any</code></td>
        <td class="s1" dir="ltr">customized</td>
        <td class="s1" dir="ltr"><code>Any</code></td>
        <td class="s2 softmerge" dir="ltr">
            <div class="softmerge-inner">Customized arguments that accessible in the instruction string</div>
        </td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">source</td>
        <td class="s1" dir="ltr"><code>str</code> or <code>list</code></td>
        <td class="s1" dir="ltr">source</td>
        <td class="s1" dir="ltr"><code>str</code></td>
        <td class="s1" dir="ltr">Source text</td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">options</td>
        <td class="s1" dir="ltr"><code>list</code></td>
        <td class="s1" dir="ltr">options</td>
        <td class="s1" dir="ltr"><code>str</code></td>
        <td class="s1" dir="ltr">Options of MCQ</td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">source_idx</td>
        <td class="s1" dir="ltr"><code>int</code></td>
        <td class="s1" dir="ltr">source_idx</td>
        <td class="s1" dir="ltr"><code>int</code></td>
        <td class="s2 softmerge" dir="ltr">
            <div class="softmerge-inner">The index of source text in options for datasets like winogrande</div>
        </td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">target</td>
        <td class="s1" dir="ltr"><code>str</code></td>
        <td class="s1" dir="ltr" rowspan="2">target</td>
        <td class="s1" dir="ltr" rowspan="2"><code>str</code></td>
        <td class="s1" dir="ltr" rowspan="2">Target text. Either return <code>target</code> or <code>target_idx</code> in <code>format_instance</code></td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">target_idx</td>
        <td class="s1" dir="ltr"><code>int</code></td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">target_idx</td>
        <td class="s1" dir="ltr"><code>int</code></td>
        <td class="s1" dir="ltr">target_idx</td>
        <td class="s1" dir="ltr"><code>int</code></td>
        <td class="s2 softmerge" dir="ltr">
            <div class="softmerge-inner">The index of target text in options for general MCQs</div>
        </td>
        </tr>
        <tr>
        <td class="s1" dir="ltr" colspan="2" rowspan="5"><i>No need to return from format_instance</i></td>
        <td class="s1" dir="ltr">example_idx</td>
        <td class="s1" dir="ltr"><code>int</code></td>
        <td class="s2 softmerge" dir="ltr">
            <div class="softmerge-inner">The index of examples if greater or equal than 0. Equal to -1 if it is not formatting an example</div>
        </td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">dataset_name</td>
        <td class="s1" dir="ltr"><code>str</code></td>
        <td class="s1" dir="ltr">Dataset name</td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">subset_name</td>
        <td class="s1" dir="ltr"><code>str</code></td>
        <td class="s2 softmerge" dir="ltr">
            <div class="softmerge-inner">Subset name of current dataset</div>
        </td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">display_name</td>
        <td class="s1" dir="ltr"><code>str</code></td>
        <td class="s2 softmerge" dir="ltr">
            <div class="softmerge-inner">Display name of current dataset</div>
        </td>
        </tr>
        <tr>
        <td class="s1" dir="ltr">real_num_shots</td>
        <td class="s1" dir="ltr"><code>int</code></td>
        <td class="s2 softmerge" dir="ltr">
            <div class="softmerge-inner">The num shots formatted as examples</div>
        </td>
        </tr>
    </tbody>
</table>

MultipleChoiceDataset:

```python
instruction = "Answer the following question.\n\n{{question}}{{'\n' + options if options}}\nAnswer:"

def format_instance(self, instance):
    return dict(
        question=instance["question"],
        target_idx=instance["answer"],
        options=[instance[label] for label in ["A", "B", "C", "D"]],
    )
```

MultipleChoiceDataset (Multiple-context) like winogrande:

```python
instruction = "Answer the following question.\n\n{source}\nAnswer:"

def format_instance(self, instance):
    return dict(
        source=[instance["ctx_a"], instance["ctx_b"]],
        source_idx=int(instance["answer"]) - 1,
        target=instance["completion"],
    )
```

GenerationDataset:

```python
instruction = "Answer the following question.\n\n{question}\nAnswer:"

def format_instance(self, instance):
    return dict(
        question=instance["question"],
        target=instance["answer"],
    )
```

To evaluate a pre-trained model that lacks instruction-following capabilities, you can provide an instruction explicitly by assigning a completion instruction to the model as follows: instruction = "{question}".

See [`Dataset`](dataset/dataset.py) for more details.

## Change Log

- **May 24, 2024**: Chat format support including conversational few-shot and system prompts.
- **May 10, 2024**: New instruction formatting using f-string and jinja2.
- **May 7, 2024**: Bump openai and vllm version.
- **Apr 16, 2024**: Full support for KV caching.
- **March 18, 2024**: First release of LLMBox.
