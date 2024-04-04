# Utilization

- [Utilization](#utilization)
  - [Usage](#usage)
    - [Model Arguments](#model-arguments)
    - [DatasetArguments](#datasetarguments)
    - [Evaluation Arguments](#evaluation-arguments)
  - [Supported Models](#supported-models)
  - [Customize Model](#customize-model)
  - [Supported Datasets](#supported-datasets)
  - [Customize Dataset](#customize-dataset)

## Usage

Evaluating davinci-002 on HellaSwag:

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
python inference.py -m llama-2-7b-hf -d cmmlu ceval --vllm True --model_type instruction
```

### Model Arguments

Specify the model, efficient evaluation, generation arguments, quantization, and other arguments.

```text
--model_name_or_path MODEL_NAME_OR_PATH, --model MODEL_NAME_OR_PATH, -m MODEL_NAME_OR_PATH
                      The model name or path, e.g., davinci-002, meta-
                      llama/Llama-2-7b-hf, ./mymodel (default: None)
--model_type {base,instruction}
                      The type of the model, which can be chosen from `base`
                      or `instruction`. (default: base)
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
                      The tokenizer name or path, e.g., meta-
                      llama/Llama-2-7b-hf (default: None)
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
--bnb_config BNB_CONFIG
                      JSON string for BitsAndBytesConfig parameters.
                      (default: None)
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
```

### DatasetArguments

Specify the dataset, batch size, example strategies, CoT strategies and other arguments.

```text
--dataset_name DATASET_NAME, -d DATASET_NAME, --dataset DATASET_NAME
                      The name of a dataset or the name(s) of a/several
                      subset(s) in a dataset. Format: 'dataset1 dataset2',
                      'dataset:subset1,subset2', or 'dataset:[cat1],[cat2]',
                      e.g., copa, race, race:high, wmt16:en-ro,en-fr, or
                      mmlu:[stem],[humanities] (default: None)
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
--system_prompt SYSTEM_PROMPT, -sys SYSTEM_PROMPT
                      The system prompt of the model (default: )
--instance_format INSTANCE_FORMAT, -fmt INSTANCE_FORMAT
                      The format to format the `source` and `target` for
                      each instance (default: {source}{target})
--num_shots NUM_SHOTS, -shots NUM_SHOTS
                      The few-shot number for demonstration (default: 0)
--ranking_type {ppl,prob,ppl_no_option}
                      The evaluation and prompting method for ranking task
                      (default: ppl_no_option)
--max_example_tokens MAX_EXAMPLE_TOKENS
                      The maximum token number of demonstration (default:
                      1024)
--batch_size BATCH_SIZE, -bsz BATCH_SIZE, -b BATCH_SIZE
                      The evaluation batch size (default: 1)
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
--dry_run [DRY_RUN]   Test the evaluation pipeline without actually calling
                      the model. (default: False)
--proxy_port PROXY_PORT
                      The port of the proxy (default: None)
--dataset_threading [DATASET_THREADING]
                      Load dataset with threading
--no_dataset_threading
                      Load dataset with threading
```

## Supported Models

<table>
  <tr>
      <td><b>Provider</b></td>
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
      <td><code>gpt-4-0125-preview</code>, <code>gpt-3.5-turbo</code></td>
      <td><code>generation</code>, <code>get_prob</code> (adapted by generation)</td>
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
      <td><code>generation</code>, <code>get_prob</code> (adapted by generation)</td>
  </tr>
  <tr>
      <td>Dashscope</td>
      <td>Generation</td>
      <td><code>qwen-turbo</code></td>
      <td><code>generation</code>, <code>get_prob</code> (adapted by generation)</td>
  </tr>
  <tr>
      <td>Anthropic</td>
      <td>Chat Completion Models</td>
      <td><code>claude-3-haiku-20240307</code></td>
      <td><code>generation</code>, <code>get_prob</code> (adapted by generation)</td>
  </tr>
  <tr>
      <td>vLLM</td>
      <td>LLM</td>
      <td><code>Llama-2-7b-hf</code></td>
      <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
  </tr>
</table>


## Customize Model

By inheriting the [`Model`](model/model.py) class, you can customize support for more models. You can implement `generation`, `get_ppl`, and `get_prob` methods to support different models. For example, you can implement the `generation` method for a new model as follows:

```python
class NewModel(Model):

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



## Supported Datasets

We currently support 53 commonly used datasets for LLMs. Each dataset may includes multiple subsets, or is a subset of a collection.

<table>
  <tr>
      <td><b>Dataset</b></td>
      <td><b>Subsets / Collections</b></td>
      <td><b>Evaluation Type</b></td>
      <td><b>CoT</b></td>
      <td><b>Notes</b></td>
  </tr>
  <tr>
      <td rowspan=3>agieval<br>(alias of <i>agieval_single_choice</i> and <i>agieval_cot</i>)</td>
      <td><b>English</b>: <code>sat-en</code>, <code>sat-math</code>, <code>lsat-ar</code>, <code>lsat-lr</code>, <code>lsat-rc</code>, <code>logiqa-en</code>, <code>aqua-rat</code>, <code>math</code></td>
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
      <td>alpaca_eval</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Single GPTEval</td>
  </tr>
  <tr>
      <td>anli</td>
      <td><code>Round2</code> (default)</td>
      <td>MultipleChoice</td>
      <td>⭕️</td>
      <td></td>
  </tr>
  <tr>
      <td>arc</td>
      <td><code>ARC-Easy</code>, <code>ARC-Challenge</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>bbh</td>
      <td><code>boolean_expressions</code>, ...</td>
      <td>Generation</td>
      <td>✅</td>
      <td></td>
  </tr>
  <tr>
      <td>boolq</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>cb</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td rowspan=4>ceval</td>
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
      <td rowspan=4>cmmlu</td>
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
      <td>cnn_dailymail</td>
      <td><code>3.0.0</code> (default), ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>color_objects</td>
      <td><i>bigbench</i> (reasoning_about_colored_objects)</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>commonsenseqa</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>copa</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>coqa</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>crows_pairs</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>drop</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td rowspan=3>gaokao</td>
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
      <td>gsm8k</td>
      <td><code>main</code> (default), <code>socratic</code></td>
      <td>Generation</td>
      <td>✅</td>
      <td>Code exec</td>
  </tr>
  <tr>
      <td>halueval</td>
      <td><code>dialogue</code>, <code>general</code>, ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>hellaswag</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>humaneval</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>ifeval</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>lambada</td>
      <td><code>default</code> (default), <code>de</code>, ... (source: <i>EleutherAI/lambada_openai</i>)</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>math</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>mbpp</td>
      <td><code>full</code> (default), <code>sanitized</code></td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td rowspan=4>mmlu</td>
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
      <td>mt_bench</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Multi-turn GPTEval</td>
  </tr>
  <tr>
      <td>nq</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>openbookqa</td>
      <td><code>main</code> (default), <code>additional</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>penguins_in_a_table</td>
      <td><i>bigbench</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>piqa</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>quac</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>race</td>
      <td><code>high</code>, <code>middle</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>real_toxicity_prompts</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Perlexity Toxicity</td>
  </tr>
  <tr>
      <td>rte</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>siqa</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>squad, squad_v2</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>story_cloze</td>
      <td><code>2016</code> (default), <code>2018</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td><a href='http://goo.gl/forms/aQz39sdDrO'>Manually download</a></td>
  </tr>
  <tr>
      <td>tldr</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>triviaqa</td>
      <td><code>rc.wikipedia.nocontext</code> (default), <code>rc</code>, <code>rc.nocontext</code>, ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>truthfulqa_mc</td>
      <td><code>multiple_choice</code> (default), <code>generation</code> (not supported)</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>vicuna_bench</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>GPTEval</td>
  </tr>
  <tr>
      <td>webq</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>wic</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>winogender</td>
      <td><code>main</code>, <code>gotcha</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Group by gender</td>
  </tr>
  <tr>
      <td>winograd</td>
      <td><code>wsc273</code> (default), <code>wsc285</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>winogrande</td>
      <td><code>winogrande_debiased</code> (default), ...</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>wmt21, wmt19, ...</td>
      <td><code>en-ro</code>, <code>ro-en</code>, ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>wsc</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>xsum</td>
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

## Customize Dataset

We provide two types of datasets: [`GenerationDataset`](dataset/generation_dataset.py), [`MultipleChoiceDataset`](dataset/multiple_choice_dataset.py). You can also customize support for a new dataset type by inheriting the [`Dataset`](dataset/dataset.py) class. For example, you can implement a new `GenerationDataset` as follows:

```python
def NewDataset(GenerationDataset):

    instruction = "Answer the following question."
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

See [`Dataset`](dataset/dataset.py) for more details.
