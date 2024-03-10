from .sftdataset import SFTDataset
import torch

class OpenAssistantDataset(SFTDataset):
    """
    OpenAssistant Conversations dataset is a curated multilingual corpus with 161,443 messages in 35 languages.

    It comprises 91,829 user prompts, 69,614 assistant replies, and 461,292 quality ratings across 66,497 conversation trees.

    Each instance represents a conversation tree (CT) where nodes denote messages by prompter or assistant.

    When executing sft, we turn the CT into some multi-turn conversations from root to every leaf node.
    """

    instruction_template = "\n\n<|prompter|>\n"
    response_template = "\n\n<|assistant|>\n"

    def process(self, tokenizer):
        input_ids = []
        labels = []
        list_data_dict = self.load_data()
        for example in list_data_dict:
            tmp1 = []
            tmp2 = []
            for s, t in zip(example['conversations'][::2], example['conversations'][1::2]):
                s = self.instruction_template + s + '\n' + self.response_template
                input_id, label = self.encode_src_tgt(s, t, tokenizer)
                tmp1.append(input_id)
                tmp2.append(label)
            input_ids.append(torch.cat(tmp1))
            labels.append(torch.cat(tmp2))
        return input_ids, labels
