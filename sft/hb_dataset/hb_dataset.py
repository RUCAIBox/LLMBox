from datasets import load_dataset,load_from_disk
import json
import torch
from typing import Dict
from itertools import chain
import os
import random
import math 

from pt_dataset import PTDataset
from sft_dataset import AutoDataset

class HBDataset:
    """
    This is the base class for all HB(Hybrid) datasets.
    """
    def __init__(self, args, tokenizer):
        self.args = args
        self.block_size = self.args.model_max_length
        self.tokenizer = tokenizer
        data_path = args.data_path
        self.data_path = args.data_path
        
        pth_file = data_path + f"_{tokenizer.model_max_length}_hybrid.pth"

        if not os.path.exists(pth_file):
            if args.dataset_list == "":
                raise ValueError(f"Cannot find the required file: {data_path}") 
            else:
                self.file_list = self.args.dataset_list.split(",")
                self.input_ids, self.labels = self.process(self.file_list)
                
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
        
    def process(self, files):
        """Process the dataset and return input_ids and labels."""
        input_ids = []
        labels = []
        training_dataset = []
        total_dataset_samples = 0

        for file_name in files:
            self.args.data_path = self.data_path + file_name

            if file_name.endswith('.txt'):
                dataset = PTDataset(self.args, self.tokenizer)
            else:
                dataset = AutoDataset(self.args, self.tokenizer)

            training_dataset.append(dataset)
            total_dataset_samples += len(dataset)

        if hasattr(self.args, "max_steps"):
            total_training_samples = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size * math.ceil(self.args.max_steps / self.args.num_train_epochs)
            if total_training_samples > total_dataset_samples:
                raise ValueError(f"Dataset could not meet the need of the max_steps and epochs.") 
        else:
            total_training_samples = total_dataset_samples

        if self.args.dataset_ratio == "":  
            for dataset in training_dataset:
                select_index = random.sample(range(len(dataset)), int(len(dataset) * total_training_samples / total_dataset_samples))
                for index in select_index:
                    input_ids.append(dataset[index]["input_ids"])
                    labels.append(dataset[index]["labels"])
        else:
            dataset_ratio = eval(self.args.dataset_ratio)
            if len(dataset_ratio) != len(training_dataset):
                raise ValueError(f"The length of the datasets and the dataset_ratio should be the same.") 
            
            for i, dataset in enumerate(training_dataset):
                if len(dataset) < int(total_training_samples * dataset_ratio[i]):
                    raise ValueError(f"Data in {dataset.args.data_path} cannot meet the requirement of the dataset length.") 

                select_index = random.sample(range(len(dataset)), int(total_training_samples * dataset_ratio[i]))
                for index in select_index:
                    input_ids.append(dataset[index]["input_ids"])
                    labels.append(dataset[index]["labels"])

        return input_ids, labels



            
                
