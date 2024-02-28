from datasets import load_dataset,load_from_disk
import json
import torch
from typing import Dict
import os
class SFTDataset:
    """
    This is the base class for all SFT datasets.
    """
    instruction_template = "\n### Instruction:\n"
    response_template = "\n### Output:\n"
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
    
    def __init__(self, args, tokenizer):
        self.args = args
        data_path = args.data_path
        
        pth_file = data_path + f"_{tokenizer.model_max_length}.pth"
        if os.path.exists(pth_file):
            checkpoint = torch.load(pth_file)
            self.input_ids = checkpoint['input_ids']
            self.labels = checkpoint['labels']
        else:
            self.input_ids ,self.labels = self.process(tokenizer)
        if torch.distributed.get_rank() == 0:
            checkpoint = {'input_ids': self.input_ids, 'labels': self.labels}
            torch.save(checkpoint, pth_file)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    
    def encode_src_tgt(self, s, t, tokenizer):
        source_id = tokenizer.encode(s, max_length=tokenizer.model_max_length, truncation=True)[:-1] # remove eos
        input_id = tokenizer.encode(s + t, max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')[0]
        label = input_id.clone()
        label[:len(source_id)] = self.args.IGNORE_INDEX
        return input_id, label
    
    def load_data(self):
        """Load data."""
        data_path = self.args.data_path
        if data_path.endswith('.jsonl'):
            list_data_dict = [json.loads(l.strip()) for l in open(data_path)]
        elif data_path.endswith('.json'):
            list_data_dict = json.load(open(data_path))
        elif os.path.isdir(data_path):
            list_data_dict = load_from_disk(data_path)['train']
        else: 
            try:
                list_data_dict = load_dataset(data_path)['train']
            except:
                raise ValueError(f"Unsupported file format: {data_path}") # TODO: Add support for other file formats
        return list_data_dict
        
    def process(self,tokenizer):
        """Process the dataset and return input_ids and labels."""
        input_ids = []
        labels = []
        list_data_dict = self.load_data()
        for example in list_data_dict:
            example['response'] = example.pop('output') # change the key name from 'output' to 'response'
            s = (self.format_template["prompt_input"].format_map(example) if 'input' in example.keys() else self.format_template["prompt_no_input"].format_map(example)).strip()
            t = example['response'].strip()
            input_id, label = self.encode_src_tgt(s, t, tokenizer)
            input_ids.append(input_id)
            labels.append(label)
        return input_ids, labels
    

