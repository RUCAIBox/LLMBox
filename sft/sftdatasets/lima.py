from datasets import load_dataset
from .sftdataset import SFTDataset


class LimaDataset(SFTDataset):
    """
    LIMA is an English instruction dataset consisting of a train set with 1K data instances.
    
    75% are sampled from three community question & answers websites (i.e., Stack Exchange, wikiHow, and the Pushshift Reddit Dataset); 
    
    20% are manually written by a set of the authors inspired by their interests; 
    
    5% are sampled from the Super-Natural Instructions dataset.
    """
    
    instruction_template = "\n\n### Human:\n"
    response_template = "\n\n### Assistant:\n"

    def concatenate_conversation(self, conversation):
        result = ""
        for i in range(0, len(conversation), 2):
            # Assuming even indices are instructions and odd indices are responses
            instruction = conversation[i]
            response = conversation[i + 1] if i + 1 < len(conversation) else ""

            # Concatenate conversation
            result += f'\n\n### Human:\n{instruction}\n\n### Assistant:\n{response}'
        return result

    def formatting_func(self, examples):
        output_texts = []
        for conversations in examples["conversations"]:
            text = self.concatenate_conversation(conversations)
            output_texts.append(text)
        return output_texts
