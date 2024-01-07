from .sftdataset import SFTDataset


class FlanDataset(SFTDataset):
    """
    Flanv2 contains "flan" (Flan 2021), "t0" (P3 excluding Flan 2021), "niv2" (Super-Natural Instructions) "cot" (several Chain-of-Thought datasets), and "dialog" (a few new dialog datasets)
    """

    instruction_template = "\n\n### Instruction:\n"
    response_template = "\n\n### Response:\n"

    def formatting_func(self, examples):
        output_texts = []
        for input_text, response in zip(examples["inputs"], examples["targets"]):
            text = self.instruction_template + input_text + self.response_template + response
            output_texts.append(text)
        return output_texts
