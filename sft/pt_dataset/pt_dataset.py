from datasets import load_dataset,load_from_disk
import json
import torch
from typing import Dict
import os

class PTDataset:
    """
    This is the base class for all PT(Pre-training) datasets.
    """

    IGNORE_INDEX = -100
    
    def __init__(self, args, tokenizer):
        self.args = args
        data_path = args.data_path
        
        #! if not the same setting, e.g. tokenizer max length changes, we need to reprocess the data, so we don't load the saved pth directly here
        pth_file = data_path + f"_{tokenizer.model_max_length}.pth"
        self.input_ids ,self.labels = self.process(tokenizer)
        if torch.distributed.get_rank() == 0:
            checkpoint = {'input_ids': self.input_ids, 'labels': self.labels}
            torch.save(checkpoint, pth_file)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def encode(self, sentence, tokenizer):
        input_id = tokenizer.encode(sentence, max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')[0]
        label = input_id.clone()
        return input_id, label
    
    def load_data(self):
        """Load data."""
        data_path = self.args.data_path
        list_data_dict = load_dataset('text', data_files=data_path)['train']
        return list_data_dict
        
    def process(self,tokenizer):
        """Process the dataset and return input_ids and labels."""
        input_ids = []
        labels = []
        list_data_dict = self.load_data()
        for example in list_data_dict:
            if len(example['text'].strip()) != 0:
                input_id, label = self.encode(example['text'].strip(), tokenizer)
                input_ids.append(input_id)
                labels.append(label)
        return input_ids, labels
    

