from .sftdataset import SFTDataset
import torch

class LimaDataset(SFTDataset):
    """
    LIMA is an English instruction dataset consisting of a train set with 1K data instances.

    75% are sampled from three community question & answers websites (i.e., Stack Exchange, wikiHow, and the Pushshift Reddit Dataset);

    20% are manually written by a set of the authors inspired by their interests;

    5% are sampled from the Super-Natural Instructions dataset.
    """

    def process(self, tokenizer):
        input_ids = []
        labels = []
        list_data_dict = self.load_data()
        for example in list_data_dict:
            tmp1 = []
            tmp2 = []
            for s, t in zip(example['conversations'][::2], example['conversations'][1::2]):
                s = tokenizer.apply_chat_template([{'role':'user','content':s}],tokenize=False)
                t = tokenizer.apply_chat_template([{'role':'assistant','content':t}],tokenize=False)
                input_id, label = self.encode_src_tgt(s, t, tokenizer)
                tmp1.append(input_id)
                tmp2.append(label)
            input_ids.append(torch.cat(tmp1))
            labels.append(torch.cat(tmp2))
        return input_ids, labels

