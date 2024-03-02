# LLMBox

## Quick start
```python
python inference.py -m davinci-002 -d copa
```
This is default to run the OpenAI davinci-002 model on the Copa dataset in a zero-shot manner.

```
  --model_name_or_path MODEL_NAME_OR_PATH, --model MODEL_NAME_OR_PATH, -m MODEL_NAME_OR_PATH
                        The model name or path, e.g., davinci-002, meta-llama/Llama-2-7b-hf,
                        ./mymodel (default: None)
  --model_type {base,instruction,None}
                        The type of the model, which can be chosen from `base` or
                        `instruction`. (default: None)
  --device_map DEVICE_MAP
                        The device map for model and data (default: auto)
  --vllm [VLLM]         Whether to use vllm (default: True)
  --no_vllm             Whether to use vllm (default: False)
  --flash_attention [FLASH_ATTENTION]
                        Whether to use flash attention (default: True)
  --no_flash_attention  Whether to use flash attention (default: False)
  --openai_api_key OPENAI_API_KEY
                        The OpenAI API key (default: None)
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH, --tokenizer TOKENIZER_NAME_OR_PATH
                        The tokenizer name or path, e.g., meta-llama/Llama-2-7b-hf (default:
                        None)
  --max_tokens MAX_TOKENS
                        The maximum number of tokens for output generation (default: None)
  --max_length MAX_LENGTH
                        The maximum number of tokens of model input sequence (default: None)
  --temperature TEMPERATURE
                        The temperature for models (default: None)
  --top_p TOP_P         The model considers the results of the tokens with top_p probability
                        mass. (default: None)
  --top_k TOP_K         The model considers the token with top_k probability. (default: None)
  --frequency_penalty FREQUENCY_PENALTY
                        Positive values penalize new tokens based on their existing frequency
                        in the generated text, vice versa. (default: None)
  --repetition_penalty REPETITION_PENALTY
                        Values>1 penalize new tokens based on their existing frequency in the
                        prompt and generated text, vice versa. (default: None)
  --presence_penalty PRESENCE_PENALTY
                        Positive values penalize new tokens based on whether they appear in
                        the generated text, vice versa. (default: None)
  --stop STOP [STOP ...]
                        List of strings that stop the generation when they are generated.
                        (default: None)
  --no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE
                        All ngrams of that size can only occur once. (default: None)
  --best_of BEST_OF, --num_beams BEST_OF
                        The beam size for beam search (default: None)
  --length_penalty LENGTH_PENALTY
                        Positive values encourage longer sequences, vice versa. Used in beam
                        search. (default: None)
  --early_stopping [EARLY_STOPPING]
                        Positive values encourage longer sequences, vice versa. Used in beam
                        search. (default: None)
  --dataset_name DATASET_NAME, -d DATASET_NAME, --dataset DATASET_NAME
                        The name of a dataset or the name(s) of a/several subset(s) in a
                        dataset. Format: 'dataset' or 'dataset:subset(s)', e.g., copa, race,
                        race:high, or wmt16:en-ro,en-fr (default: None)
  --dataset_path DATASET_PATH
                        The path of dataset if loading from local. Supports repository cloned
                        from huggingface or dataset saved by `save_to_disk`. (default: None)
  --evaluation_set EVALUATION_SET
                        The set name for evaluation, supporting slice, e.g., validation,
                        test, validation[:10] (default: None)
  --example_set EXAMPLE_SET
                        The set name for demonstration, supporting slice, e.g., train, dev,
                        train[:10] (default: None)
  --system_prompt SYSTEM_PROMPT, -sys SYSTEM_PROMPT
                        The system prompt of the model (default: )
  --instance_format INSTANCE_FORMAT, -fmt INSTANCE_FORMAT
                        The format to format the `source` and `target` for each instance
                        (default: {source}{target})
  --num_shots NUM_SHOTS, -shots NUM_SHOTS
                        The few-shot number for demonstration (default: 0)
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
  --cot {none,base,least_to_most,pal}
                        The method to prompt, eg. 'none', 'base', 'least_to_most', 'pal'. Only
                        available for some specific datasets. (default: none)
  --seed SEED           The random seed (default: 2023)
  --logging_dir LOGGING_DIR
                        The logging directory (default: logs)
  --log_level {debug,info,warning,error,critical}
                        Logger level to use on the main node. Possible choices are the log
                        levels as strings: 'debug', 'info', 'warning', 'error' and 'critical'
                        (default: info)
  --evaluation_results_dir EVALUATION_RESULTS_DIR
                        The directory to save evaluation results, which includes source and
                        target texts, generated texts, and the references. (default:
                        evaluation_results)
```
