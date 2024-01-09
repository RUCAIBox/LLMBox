from .sftdataset import SFTDataset


class EvolInstructDataset(SFTDataset):
    """
    Evol-instruct dataset is produced through iterative application of the Evol-Instruct technique on the Code Alpaca dataset.
    """

    instruction_template = "\n\n### Human:\n"
    response_template = "\n\n### Assistant:\n"

    def concatenate_conversation(self, conversation):
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

    def formatting_func(self, examples):
        output_texts = []
        for conversations in examples["conversations"]:
            text = self.concatenate_conversation(conversations)
            output_texts.append(text)
        return output_texts
