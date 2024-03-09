from .sftdataset import SFTDataset
import torch

class EvolInstructDataset(SFTDataset):
    """
    Evol-instruct dataset is produced through iterative application of the Evol-Instruct technique on the Code Alpaca dataset.
    """

    instruction_template = "\n\n### Human:\n"
    response_template = "\n\n### Assistant:\n"

    def process(self, tokenizer):
        input_ids = []
        labels = []
        list_data_dict = self.load_data()
        for example in list_data_dict:
            tmp1 = []
            tmp2 = []
            for s, t in zip(example['conversations'][::2], example['conversations'][1::2]):
                s = self.instruction_template + s['value'] + '\n' + self.response_template
                t = t['value']
                input_id, label = self.encode_src_tgt(s, t, tokenizer)
                tmp1.append(input_id)
                tmp2.append(label)
            input_ids.append(torch.cat(tmp1))
            labels.append(torch.cat(tmp2))
        return input_ids, labels
