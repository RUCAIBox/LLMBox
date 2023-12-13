from datasets import load_dataset


class SFTDataset():
    """
    This is the base class for all SFT datasets.
    
    Please inherit this class and implement your own `instruction_template, response_template, format_template.` and `formatting_func()`.
    
    The `context_token` is set for some special tokenizer like llama, which changes depending on the context. https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate
    
    Please refer to https://huggingface.co/docs/trl/sft_trainer for detailed instructions.
    """

    instruction_template = "\n### Instruction:\n"
    response_template = "\n### Output:\n"
    format_template = "You are a helpful assistant.\n### Instruction:\n{instruction}\n### Output:\n{output}"

    def __init__(self, args) -> None:
        self.args = args

    def load_data(self):
        if self.args.data_path.endswith(".csv"):
            return load_dataset("csv", data_files=self.args.data_path, split="train")
        elif self.args.data_path.endswith(".json") or self.args.data_path.endswith(".jsonl"):
            return load_dataset("json", data_files=self.args.data_path, split="train")
        else:
            return load_dataset(self.args.data_path, split="train")

    def formatting_func(self, examples):
        output_texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = self.format_template.format(instruction=instruction, output=output)
            output_texts.append(text)
        return output_texts
