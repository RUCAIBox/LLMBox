# How to Load Dataset GPQA

## Step 1: Install LLMBox

Install the LLMBox library by following the instructions in the [installation guide](https://github.com/RUCAIBox/LLMBox).

## Step 2: Generate a Hugging Face Token

GPQA is a gated dataset, so you will need to apply for access to the dataset with your Hugging Face account. Then, you need to generate a Hugging Face [token](https://huggingface.co/settings/tokens) to access the dataset.

> [!TIP]
> It's recommended to use a `READ`-only token.

Then you can access to the dataset by using the following command:

```bash
cd LLMBox
python inference.py -m model -d gpqa --hf_username <username> --hf_token <token>
```

Alternatively, you can store it in a `.env` file in the root of the repository.

```text
HF_TOKEN=<token>
```

## Step 3: Clone the GPQA Repository

Then you need to clone the GPQA repository to access the chain-of-thought prompts.

```bash
git clone https://github.com/idavidrein/gpqa
```

## Step 4: Evaluate on GPQA (0-shot)

```bash
python inference.py -m model -d gpqa
```

## Step 5: Evaluate on GPQA (5-shot)

```bash
python inference.py -m model -d gpqa --example_set ./gpqa --num_shots 5
```

## Step 6: Evaluate on GPQA (0-shot, CoT)

```bash
python inference.py -m model -d gpqa --example_set ./gpqa --cot base
```

## Step 7: Evaluate on GPQA (5-shot, CoT)

```bash
python inference.py -m model -d gpqa --example_set ./gpqa --num_shots 5 --cot base
```
