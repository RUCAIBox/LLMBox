from typing import List

import torch
from transformers import StoppingCriteria


class KeyWordsCriteria(StoppingCriteria):

    def __init__(self, stop_id_sequences: List[List[int]]):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences
        self.sequences_should_be_stopped = None

    def step(self):
        self.sequences_should_be_stopped = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.sequences_should_be_stopped is None:
            self.sequences_should_be_stopped = [False] * input_ids.shape[0]
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i, -len(stop_sequence):].tolist() == stop_sequence:
                    self.sequences_should_be_stopped[i] = True
                    break
        return all(self.sequences_should_be_stopped)
