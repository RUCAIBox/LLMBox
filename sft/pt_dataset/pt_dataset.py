from datasets import load_dataset,load_from_disk
import json
import torch
from typing import Dict
from itertools import chain
import os

class PTDataset:
    """
    This is the base class for all PT(Pre-training) datasets.
    """
    def __init__(self, args, tokenizer):
        self.args = args
        self.block_size = self.args.model_max_length
        self.tokenizer = tokenizer
        data_path = args.data_path
        
        #! if not the same setting, e.g. tokenizer max length changes, we need to reprocess the data, so we don't load the saved pth directly here
        pth_file = data_path + f"_{tokenizer.model_max_length}.pth"

        if not os.path.exists(pth_file):
            self.input_ids, self.labels = self.process()
            if self.args.packing:
                self.input_ids = self.group_texts(self.input_ids)
                self.labels = self.input_ids.copy()

            if torch.distributed.get_rank() == 0:
                checkpoint = {'input_ids': self.input_ids, 'labels': self.labels}
                torch.save(checkpoint, pth_file)
        else:
            data_dict = torch.load(pth_file)
            self.input_ids = data_dict['input_ids']
            self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def encode(self, examples):
        output = self.tokenizer(examples["text"])
        return output

    def group_texts(self, examples):
        """concatenate and pack the dataset"""
        concatenated_examples = list(chain(*examples))
        total_length = len(concatenated_examples)

        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size

        result = [ torch.stack(concatenated_examples[i : i + self.block_size]) for i in range(0, total_length, self.block_size) ]

        return result
    
    def load_data(self):
        """Load data."""
        data_path = self.args.data_path
        if data_path.endswith('.txt'):
            list_data_dict = load_dataset('text', data_files=data_path)['train']
        elif os.path.isdir(data_path):
            list_data_dict = load_from_disk(data_path)['train']
        else: 
            try:
                list_data_dict = load_dataset(data_path)['train']
            except:
                raise ValueError(f"Unsupported file format: {data_path}") # TODO: Add support for other file formats
        return list_data_dict
        
    def process(self):
        """Process the dataset and return input_ids and labels."""
        input_ids = []
        list_data_dict = self.load_data()

        tokenized_dataset = list_data_dict.map(
            self.encode,
            batched=True,
            remove_columns='text',
        )

        for example in tokenized_dataset:
            if len(example['input_ids']) > 0:
                input_ids.append(torch.tensor(example['input_ids']))

        return input_ids, input_ids.copy()