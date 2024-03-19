from .sftdataset import SFTDataset


class FlanDataset(SFTDataset):
    """
    Flanv2 contains "flan" (Flan 2021), "t0" (P3 excluding Flan 2021), "niv2" (Super-Natural Instructions) "cot" (several Chain-of-Thought datasets), and "dialog" (a few new dialog datasets)
    """

    instruction_template = "\n\n### Instruction:\n"
    response_template = "\n\n### Response:\n"

    def process(self,tokenizer):
        """Process the dataset and return input_ids and labels."""
        input_ids = []
        labels = []
        list_data_dict = self.load_data()
        for example in list_data_dict:
            s = self.instruction_template + example['inputs']  + self.response_template 
            t = example['targets'].strip()
            input_id, label = self.encode_src_tgt(s, t, tokenizer)
            input_ids.append(input_id)
            labels.append(label)
        return input_ids, labels
