# LLMBox

## Quick start
```python
python inference.py
```
This is default to run the OpenAI curie model on the Copa dataset in a zero-shot manner.

```
usage: inference.py [-h] [-m MODEL] [-d DATASET] [-bsz BATCH_SIZE] [--evaluation_set EVALUATION_SET] [--seed SEED] [-sys SYSTEM_PROMPT] [-format INSTANCE_FORMAT] [--example_set EXAMPLE_SET]
                    [-shots NUM_SHOTS] [--max_example_tokens MAX_EXAMPLE_TOKENS] [-api OPENAI_API_KEY]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The model name, e.g., cuire, llama
  -d DATASET, --dataset DATASET
                        The model name, e.g., copa, gsm
  -bsz BATCH_SIZE, --batch_size BATCH_SIZE
                        The evaluation batch size
  --evaluation_set EVALUATION_SET
                        The set name for evaluation
  --seed SEED           The random seed
  -sys SYSTEM_PROMPT, --system_prompt SYSTEM_PROMPT
                        The system prompt of the model
  -format INSTANCE_FORMAT, --instance_format INSTANCE_FORMAT
                        The format to format the `source` and `target` for each instance
  --example_set EXAMPLE_SET
                        The set name for demonstration
  -shots NUM_SHOTS, --num_shots NUM_SHOTS
                        The few-shot number for demonstration
  --max_example_tokens MAX_EXAMPLE_TOKENS
                        The maximum token number of demonstration
  -api OPENAI_API_KEY, --openai_api_key OPENAI_API_KEY
                        The OpenAI API key
```
