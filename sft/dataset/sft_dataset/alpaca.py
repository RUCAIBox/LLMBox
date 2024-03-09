from .sftdataset import SFTDataset


class AlpacaDataset(SFTDataset):
    """
    Stanford alpaca's dataset is a 52K instruction-following demonstrations generated from OpenAI’s text-davinci-003.
    """

    instruction_template = "\n\n### Instruction:\n"
    response_template = "\n\n### Response:\n"
    format_template = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
            "{input}" + response_template
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
            response_template 
        ),
    }
    
    def process(self,tokenizer):
        """Process the dataset and return input_ids and labels."""
        input_ids = []
        labels = []
        list_data_dict = self.load_data()
        for example in list_data_dict:
            example['response'] = example.pop('output') # change the key name from 'output' to 'response'
            s = (self.format_template["prompt_input"].format_map(example) if 'input' in example else self.format_template["prompt_no_input"].format_map(example)).strip()
            t = example['response'].strip()
            input_id, label = self.encode_src_tgt(s, t, tokenizer)
            input_ids.append(input_id)
            labels.append(label)
        return input_ids, labels
    
