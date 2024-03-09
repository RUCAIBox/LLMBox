from datasets import load_dataset,load_from_disk
import json
import torch
from typing import Dict
from itertools import chain
import os
import random
import math 
import bisect

from collections import OrderedDict

from .sft_dataset.alpaca import AlpacaDataset
from .sft_dataset.belle import BelleDataset
from .sft_dataset.dolly import DollyDataset
from .sft_dataset.evol_instruct import EvolInstructDataset
from .sft_dataset.flan import FlanDataset
from .sft_dataset.lima import LimaDataset
from .sft_dataset.openassistant import OpenAssistantDataset
from .sft_dataset.self_instruct import SelfInstructDataset
from .sft_dataset.sharegpt import ShareGPTDataset

# You can add your own dataset name and corresponding class here
DATASETNAMEMAP = OrderedDict({
    "alpaca": AlpacaDataset,
    "belle": BelleDataset,
    "self_instruct": SelfInstructDataset,
    "evol_instruct": EvolInstructDataset,
    "dolly": DollyDataset,
    "lima": LimaDataset,
    "sharegpt": ShareGPTDataset,
    "openassistant": OpenAssistantDataset,
    "flan": FlanDataset,
})

class BaseDataset:
    """
    This is the base class for all datasets.
    """
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        data_path = args.data_path
        self.data_path = args.data_path
        self.total_training_samples = 0
        self.dataset_start_index = [0]
        self.training_dataset = []
        
        if args.dataset_list == "":
            raise ValueError(f"Cannot find the required file: {data_path}") 
        else:
            self.process(self.args.dataset_list)

    def __len__(self):
        return self.total_training_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        dataset_index = bisect.bisect_right(self.dataset_start_index, i) - 1
        return self.training_dataset[dataset_index][(i - self.dataset_start_index[dataset_index]) % len(self.training_dataset[dataset_index])]
   
    def process(self, files):
        for file_name in files:
            self.args.data_path = self.data_path + file_name

            if file_name.endswith('.txt'):
                dataset = PTDataset(self.args, self.tokenizer)
            else:
                dataset = SFTDataset(self.args, self.tokenizer)

            self.training_dataset.append(dataset)

        if self.args.dataset_ratio is None:
            # Concat all the provided dataset as the new hybrid dataset
            for i, dataset in enumerate(self.training_dataset):
                self.total_training_samples += len(dataset)
                if i > 0:
                    self.dataset_start_index.append(self.dataset_start_index[-1] + len(dataset))
        else:
            if hasattr(self.args, "max_steps"):
                max_total_samples = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size * self.args.max_steps
                
                dataset_ratio = self.args.dataset_ratio
                if len(dataset_ratio) != len(self.training_dataset):
                    raise ValueError(f"The length of the datasets and the dataset_ratio should be the same.") 

                for i, dataset in enumerate(self.training_dataset):
                    self.total_training_samples += int(max_total_samples * dataset_ratio[i])
                    if i > 0:
                        self.dataset_start_index.append(self.dataset_start_index[-1] + int(max_total_samples * dataset_ratio[i]))
            else:
                raise ValueError(f"Max_steps must be set when dataset_ratio is set.") 


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

        self.shuffle_index = random.sample(range(len(self.input_ids)), len(self.input_ids))

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

class SFTDataset:
    def __new__(self, args, tokenizer):
        datapath = args.data_path
        for datasetname, datasetclass in DATASETNAMEMAP.items():
            # if the datasetname is in the datapath, then we select this dataset
            if datasetname in datapath:
                import warnings
                warnings.warn(f"Dataset: {datasetname} is selected", stacklevel=2)
                return datasetclass(args, tokenizer)

        # failed to find the dataset
        raise ValueError(
            f"Your {datapath} should contain names like these: {DATASETNAMEMAP.keys()}, so that it can find our sftdataset class. Or you can add your own dataset class."
        )