from datasets import load_dataset,load_from_disk
import json
import torch
from typing import Dict
from itertools import chain
import os
import random

class PTDataset:
    """
    This is the base class for all PT(Pre-training) datasets.
    """
    def __init__(self, args, tokenizer):
        self.args = args
        self.block_size = self.args.model_max_length
        self.tokenizer = tokenizer
        data_path = args.data_path
        
        pth_file = f"{data_path}.pth"
        retokenize = False
        if os.path.exists(pth_file):
            data_dict = torch.load(pth_file)
            prev_tokenizer_config = data_dict['tokenizer_config']
            current_tokenizer_config = str(tokenizer)
            if prev_tokenizer_config != current_tokenizer_config:
                retokenize = True
            else:
                print("Loading tokenized data from cache")
                self.input_ids = data_dict['input_ids']
                self.labels = data_dict['labels']
        else:
            retokenize = True

        if retokenize:
            self.input_ids, self.labels = self.process(self.tokenizer)
            if self.args.packing:
                self.input_ids, self.labels = self.group_texts(self.input_ids)

            if torch.distributed.get_rank() == 0:
                checkpoint = {'input_ids': self.input_ids, 'labels': self.labels, 'tokenizer_config': str(tokenizer)}
                torch.save(checkpoint, pth_file)


        self.shuffle_index = list(range(len(self.input_ids)))
        random.shuffle(self.shuffle_index)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        index = self.shuffle_index[i]
        return dict(input_ids=self.input_ids[index], labels=self.labels[index])
    
    def encode(self, examples):
        output = self.tokenizer(examples["text"], truncation=True)
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