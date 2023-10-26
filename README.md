# LLMBox

## Quick start
```
usage: main.py [-m MODEL] [-d DATASET] [-bsz BATCH_SIZE] [--evaluation_set EVALUATION_SET] [--seed SEED] [-inst INSTRUCTION] [--example_set EXAMPLE_SET] [-shots NUM_SHOTS]
               [--max_example_tokens MAX_EXAMPLE_TOKENS] [--example_separator_string EXAMPLE_SEPARATOR_STRING] [-api OPENAI_API_KEY]

optional arguments:
  -m MODEL, --model MODEL
                        The model name, e.g., cuire, llama
  -d DATASET, --dataset DATASET
                        The model name, e.g., copa, gsm
  -bsz BATCH_SIZE, --batch_size BATCH_SIZE
                        The evaluation batch size
  --evaluation_set EVALUATION_SET
                        The set name for evaluation
  --seed SEED           The random seed
  -inst INSTRUCTION, --instruction INSTRUCTION
                        The instruction to format each instance
  --example_set EXAMPLE_SET
                        The set name for demonstration
  -shots NUM_SHOTS, --num_shots NUM_SHOTS
                        The few-shot number for demonstration
  --max_example_tokens MAX_EXAMPLE_TOKENS
                        The maximum token number of demonstration
  --example_separator_string EXAMPLE_SEPARATOR_STRING
                        The string to separate each demonstration
  -api OPENAI_API_KEY, --openai_api_key OPENAI_API_KEY
                        The OpenAI API key
```
