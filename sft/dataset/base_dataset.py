from datasets import load_dataset,load_from_disk
import json
import torch
from typing import Dict
from itertools import chain
import os
import random
import math 
import bisect

from .sft_dataset import SFTDataset
from .pt_dataset import PTDataset

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
        
        if args.dataset == "":
            raise ValueError(f"Cannot find the required file: {data_path}") 
        else:
            self.process(self.args.dataset)

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
            if not hasattr(self.args, "max_steps"):
                raise ValueError(f"Max_steps must be set when dataset_ratio is set.") 

            if len(self.args.dataset_ratio) != len(self.training_dataset):
                raise ValueError(f"The length of the datasets and the dataset_ratio should be the same.") 

            max_total_samples = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size * self.args.max_steps
            for i, dataset in enumerate(self.training_dataset):
                self.total_training_samples += int(max_total_samples * self.args.dataset_ratio[i])
                if i > 0:
                    self.dataset_start_index.append(self.dataset_start_index[-1] + int(max_total_samples * self.args.dataset_ratio[i]))