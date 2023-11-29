from datasets import load_dataset
from collections import OrderedDict
# TODO check the format when calling "load_dataset", it may failed
# Base dataset for sft
class SFTDataset():
    instruction_template = "\n\n### Instruction:\n"
    response_template = "\n\n### Output:\n"
    format_template = "You are a helpful assistant.\n\n### Instruction:\n{instruction}\n\n### Output:\n{output}"
    
    def __init__(self,args) -> None:
        self.args = args
    
    def load_data(self):
        if self.args.data_path.endswith(".csv"):
            return load_dataset('csv', data_files=self.args.data_path)['train']
        else:
            return load_dataset('json', data_files=self.args.data_path)['train']
    
    def formatting_func(self,examples):
        output_texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = self.format_template.format(instruction=instruction, output=output)
            output_texts.append(text)
        return output_texts

class AlpacaDataset(SFTDataset):
    instruction_template = "\n\n### Instruction:\n"
    response_template = "\n\n### Response:\n"
    format_template = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{response}"
            ),
        }
    def formatting_func(self,examples):
        output_texts = []
        for instruction, input_text, response in zip(examples["instruction"], examples["input"], examples["output"]):
            if input_text:
                text = self.format_template["prompt_input"].format(instruction=instruction, input=input_text, response=response)
            else:
                text = self.format_template["prompt_no_input"].format(instruction=instruction, response=response)
            output_texts.append(text)
        return output_texts

class SelfInstructDataset(SFTDataset):
    instruction_template = "\n\n### Instruction:\n"
    response_template = "\n\n### Response:\n"
    format_template = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{response}"
            ),
        }
    def formatting_func(self,examples):
        output_texts = []
        for instruction, input_text, response in zip(examples["instruction"], examples["input"], examples["output"]):
            if input_text:
                text = self.format_template["prompt_input"].format(instruction=instruction, input=input_text, response=response)
            else:
                text = self.format_template["prompt_no_input"].format(instruction=instruction, response=response)
            output_texts.append(text)
        return output_texts
    

class EvolInstructDataset(SFTDataset):
    """
    Chat form
    """
    instruction_template = "\n\n### Human:\n"
    response_template = "\n\n### Assistant:\n"
    
    def concatenate_conversation(self,conversation):
        result = ""
        for message in conversation:
            sender = message['from']
            value = message['value']

            # Concatenate the sender and message value
            if sender == "human":
                result += f"\n\n### Human:\n{value}"
            elif sender == "gpt":
                result += f"\n\n### Assistant:\n{value}"

        return result
    def formatting_func(self,examples):
        output_texts = []
        for conversations in examples["conversations"]:
            text = self.concatenate_conversation(conversations)
            output_texts.append(text)
        return output_texts
    
class DollyDataset(SFTDataset):
    instruction_template = "\n\n### Instruction:\n"
    response_template = "\n\n### Response:\n"
    format_template = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{response}"
            ),
        }
    
    def formatting_func(self,examples):
        output_texts = []
        for instruction, input_text, response in zip(examples["instruction"], examples["context"], examples["output"]):
            if input_text:
                text = self.format_template["prompt_input"].format(instruction=instruction, input=input_text, response=response)
            else:
                text = self.format_template["prompt_no_input"].format(instruction=instruction, response=response)
            output_texts.append(text)
        return output_texts

class LimaDataset(SFTDataset):
    """
    Chat form
    """
    instruction_template = "\n\n### Human:\n"
    response_template = "\n\n### Assistant:\n"
        
    def concatenate_conversation(self,conversation):
        result = ""
        for i in range(0, len(conversation), 2):
            # Assuming even indices are instructions and odd indices are responses
            instruction = conversation[i]
            response = conversation[i + 1] if i + 1 < len(conversation) else ""

            # Concatenate conversation
            result += f'\n\n### Human:\n{instruction}\n\n### Assistant:\n{response}'
        return result
    
    def formatting_func(self,examples):
        output_texts = []
        for conversations in examples["conversations"]:
            text = self.concatenate_conversation(conversations)
            output_texts.append(text)
        return output_texts
    

# You can add your own dataset name and corresponding class here
DATASETNAMEMAP = OrderedDict({
    "alpaca": AlpacaDataset,
    "selfinstruct": SelfInstructDataset,
    "evolinstruct": EvolInstructDataset,
    "dolly": DollyDataset,
    "lima": LimaDataset,
})

class AutoDataset():
    def __new__(self,args):
        datapath = args.data_path
        for datasetname, datasetclass in DATASETNAMEMAP.items():
            # if the datasetname is in the datapath, then we select this dataset
            if datasetname in datapath:
                print(f"AutoDataset: {datasetname} is selected")
                return datasetclass(args)
                
        # failed to find the dataset
        raise ValueError(f"AutoDataset: not supported dataset name: {datapath}, leagal names are: {DATASETNAMEMAP.keys()}, or you can add your own dataset class to {__file__}.")
        
        