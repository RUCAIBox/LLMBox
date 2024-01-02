from .sft_dataset import SFTDataset
from datasets import load_dataset


class FlanDataset(SFTDataset):
    """
    Flanv2 contains "flan" (Flan 2021), "t0" (P3 excluding Flan 2021), "niv2" (Super-Natural Instructions) "cot" (several Chain-of-Thought datasets), and "dialog" (a few new dialog datasets)
    """

    instruction_template = "\n\n### Instruction:\n"
    response_template = "\n\n### Response:\n"
    format_template = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" + "{input}" +
            response_template + "{response}"
        ),
    }

    def formatting_func(self, examples):
        output_texts = []
        for input_text, response in zip(examples["inputs"], examples["targets"]): # there is no fixed instruction in this dataset
            text = self.format_template["prompt_input"].format(instruction='',input=input_text, response=response)
            output_texts.append(text)
        return output_texts