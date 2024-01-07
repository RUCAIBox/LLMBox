from .sftdataset import SFTDataset


class OpenAssistantDataset(SFTDataset):
    """
    OpenAssistant Conversations dataset is a curated multilingual corpus with 161,443 messages in 35 languages. 
    
    It comprises 91,829 user prompts, 69,614 assistant replies, and 461,292 quality ratings across 66,497 conversation trees. 
    
    Each instance represents a conversation tree (CT) where nodes denote messages by prompter or assistant. 
    
    When executing sft, we turn the CT into some multi-turn conversations from root to every leaf node.
    """

    instruction_template = "\n\n<|prompter|>\n"
    response_template = "\n\n<|assistant|>\n"

    def concatenate_conversation(self, conversation):
        result = ""
        for i in range(0, len(conversation), 2):
            # Assuming even indices are instructions and odd indices are responses
            instruction = conversation[i]
            response = conversation[i + 1] if i + 1 < len(conversation) else ""

            # Concatenate conversation
            result += f'{self.instruction_template}{instruction}{self.response_template}{response}'
        return result

    def formatting_func(self, examples):
        output_texts = []
        for conversations in examples["conversations"]:
            text = self.concatenate_conversation(conversations)
            output_texts.append(text)
        return output_texts
