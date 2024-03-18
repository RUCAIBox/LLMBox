# Utilization

## Table of contents

- [Utilization](#utilization)
  - [Table of contents](#table-of-contents)
  - [Usage](#usage)
  - [Supported Models](#supported-models)
  - [Supported Datasets](#supported-datasets)

## Usage

```text
  --model_name_or_path MODEL_NAME_OR_PATH, --model MODEL_NAME_OR_PATH, -m MODEL_NAME_OR_PATH
                        The model name or path, e.g., davinci-002, meta-llama/Llama-2-7b-hf, ./mymodel (default:
                        None)
  --model_type {base,instruction}
                        The type of the model, which can be chosen from `base` or `instruction`. (default: None)
  --device_map DEVICE_MAP
                        The device map for model and data (default: auto)
  --vllm [VLLM]         Whether to use vllm (default: None)
  --flash_attention [FLASH_ATTENTION]
                        Whether to use flash attention (default: None)
  --prefix_caching [PREFIX_CACHING]
                        Whether to cache prefix in get_ppl mode (default: None)
  --openai_api_key OPENAI_API_KEY
                        The OpenAI API key (default: None)
  --anthropic_api_key ANTHROPIC_API_KEY
                        The Anthropic API key (default: None)
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH, --tokenizer TOKENIZER_NAME_OR_PATH
                        The tokenizer name or path, e.g., meta-llama/Llama-2-7b-hf (default: None)
  --max_tokens MAX_TOKENS
                        The maximum number of tokens for output generation (default: None)
  --max_length MAX_LENGTH
                        The maximum number of tokens of model input sequence (default: None)
  --temperature TEMPERATURE
                        The temperature for models (default: None)
  --top_p TOP_P         The model considers the results of the tokens with top_p probability mass. (default: None)
  --top_k TOP_K         The model considers the token with top_k probability. (default: None)
  --frequency_penalty FREQUENCY_PENALTY
                        Positive values penalize new tokens based on their existing frequency in the generated
                        text, vice versa. (default: None)
  --repetition_penalty REPETITION_PENALTY
                        Values>1 penalize new tokens based on their existing frequency in the prompt and generated
                        text, vice versa. (default: None)
  --presence_penalty PRESENCE_PENALTY
                        Positive values penalize new tokens based on whether they appear in the generated text,
                        vice versa. (default: None)
  --stop STOP [STOP ...]
                        List of strings that stop the generation when they are generated. E.g. --stop 'stop'
                        'sequence' (default: None)
  --no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE
                        All ngrams of that size can only occur once. (default: None)
  --best_of BEST_OF, --num_beams BEST_OF
                        The beam size for beam search (default: None)
  --length_penalty LENGTH_PENALTY
                        Positive values encourage longer sequences, vice versa. Used in beam search. (default:
                        None)
  --early_stopping [EARLY_STOPPING]
                        Positive values encourage longer sequences, vice versa. Used in beam search. (default:
                        None)
  --bnb_config BNB_CONFIG
                        JSON string for BitsAndBytesConfig parameters. (default: None)
  --load_in_8bit [LOAD_IN_8BIT]
                        Whether to use bnb's 8-bit quantization to load the model. (default: False)
  --load_in_4bit [LOAD_IN_4BIT]
                        Whether to use bnb's 4-bit quantization to load the model. (default: False)
  --gptq [GPTQ]         Whether the model is a gptq quantized model. (default: False)
  --dataset_name DATASET_NAME, -d DATASET_NAME, --dataset DATASET_NAME
                        The name of a dataset or the name(s) of a/several subset(s) in a dataset. Format: 'dataset'
                        or 'dataset:subset(s)', e.g., copa, race, race:high, or wmt16:en-ro,en-fr (default: None)
  --dataset_path DATASET_PATH
                        The path of dataset if loading from local. Supports repository cloned from huggingface,
                        dataset saved by `save_to_disk`, or a template string e.g.
                        'mmlu/{split}/{subset}_{split}.csv'. (default: None)
  --evaluation_set EVALUATION_SET
                        The set name for evaluation, supporting slice, e.g., validation, test, validation[:10]
                        (default: None)
  --example_set EXAMPLE_SET
                        The set name for demonstration, supporting slice, e.g., train, dev, train[:10] (default:
                        None)
  --system_prompt SYSTEM_PROMPT, -sys SYSTEM_PROMPT
                        The system prompt of the model (default: )
  --instance_format INSTANCE_FORMAT, -fmt INSTANCE_FORMAT
                        The format to format the `source` and `target` for each instance (default:
                        {source}{target})
  --num_shots NUM_SHOTS, -shots NUM_SHOTS
                        The few-shot number for demonstration (default: 0)
  --ranking_type {ppl,prob,ppl_no_option}
                        The evaluation and prompting method for ranking task (default: ppl_no_option)
  --max_example_tokens MAX_EXAMPLE_TOKENS
                        The maximum token number of demonstration (default: 1024)
  --batch_size BATCH_SIZE, -bsz BATCH_SIZE, -b BATCH_SIZE
                        The evaluation batch size (default: 1)
  --sample_num SAMPLE_NUM, --majority SAMPLE_NUM, --consistency SAMPLE_NUM
                        The sampling number for self-consistency (default: 1)
  --kate [KATE], -kate [KATE]
                        Whether to use KATE as an ICL strategy (default: False)
  --globale [GLOBALE], -globale [GLOBALE]
                        Whether to use GlobalE as an ICL strategy (default: False)
  --ape [APE], -ape [APE]
                        Whether to use APE as an ICL strategy (default: False)
  --cot {None,base,least_to_most,pal}
                        The method to prompt, eg. 'none', 'base', 'least_to_most', 'pal'. Only available for some
                        specific datasets. (default: None)
  --perspective_api_key PERSPECTIVE_API_KEY
                        The Perspective API key (default: None)
  --proxy_port PROXY_PORT
                        The port of the proxy (default: None)
  --pass_at_k PASS_AT_K
                        The k value for pass@k metric (default: None)
  --seed SEED           The random seed (default: 2023)
  --logging_dir LOGGING_DIR
                        The logging directory (default: logs)
  --log_level {debug,info,warning,error,critical}
                        Logger level to use on the main node. Possible choices are the log levels as strings:
                        'debug', 'info', 'warning', 'error' and 'critical' (default: info)
  --evaluation_results_dir EVALUATION_RESULTS_DIR
                        The directory to save evaluation results, which includes source and target texts, generated
                        texts, and the references. (default: evaluation_results)
  --dry_run [DRY_RUN]   Test the evaluation pipeline without actually calling the model. (default: False)
```

## Supported Models

<table>
    <tr>
        <td><b>Provider</b></td>
        <td><b>Model</b></td>
        <td><b>Supported Methods</b></td>
    </tr>
    <tr>
        <td>Huggingface</td>
        <td>AutoModelForCasualLM</td>
        <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
    </tr>
    <tr>
        <td rowspan=2>OpenAI</td>
        <td>Chat Completion Models</td>
        <td><code>generation</code>, <code>get_prob</code> (adapted by generation)</td>
    </tr>
    <tr>
        <td>Completion Models (Legacy)</td>
        <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
    </tr>
    <tr>
        <td>Anthropic</td>
        <td>Chat Completion Models</td>
        <td><code>generation</code>, <code>get_prob</code> (adapted by generation)</td>
    </tr>
    <tr>
        <td>vLLM</td>
        <td>LLM</td>
        <td><code>generation</code>, <code>get_ppl</code>, <code>get_prob</code></td>
    </tr>
</table>


## Supported Datasets
<table>
    <tr>
        <td><b>Dataset</b></td>
        <td><b>Subsets</b></td>
        <td><b>Evaluation Type</b></td>
        <td><b>CoT</b></td>
        <td><b>Additional Processor</b></td>
        <td><b>Reference</b></td>
    </tr>
    <tr>
        <td rowspan=2>agieval</td>
        <td>English: sat-en,sat-math,lsat-ar,lsat-lr,lsat-rc,logiqa-en,aqua-rat,math</td>
        <td rowspan=2>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Chinese: gaokao-chinese, gaokao-geography, gaokao-history, gaokao-biology, gaokao-chemistry, gaokao-physics, gaokao-mathqa</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>alpaca_eval</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>anli</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>arc</td>
        <td>ARC-Easy, ARC-Challenge</td>
        <td>MultipleChoice</td>
        <td>No</td>
        <td>Normalization</td>
        <td></td>
    </tr>
    <tr>
        <td>bbh</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>boolq</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>cb</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ceval</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>cmmlu</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>cnn_dailymail</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>color_objects</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>commonsenseqa</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>copa</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>coqa</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>crows_pairs</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>drop</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>gaokao</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>gsm8k</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>halueval</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>hellaswag</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>humaneval</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ifeval</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>lambada</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>math</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>mbpp</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>mmlu</td>
        <td></td>
        <td>MultipleChoice</td>
        <td>✅</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>mt_bench</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>nq</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>openbookqa</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>penguins_in_a_table</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>piqa</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>quac</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>race</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>real_toxicity_prompts</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>rte</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>siqa</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>squad</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>story_cloze</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>tldr</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>translation</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>triviaqa</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>truthfulqa_mc</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>vicuna_bench</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>webq</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>wic</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>winogender</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>winograd</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>winogrande</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>wsc</td>
        <td></td>
        <td>MultipleChoice</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>xsum</td>
        <td></td>
        <td>Generation</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>

</table>
