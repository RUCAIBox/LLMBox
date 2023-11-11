from .math_word_dataset import MathWordDataset
from datasets import load_dataset, load_from_disk
import re


class Gsm8k(MathWordDataset):
    """The dataset of GSM8K.

    GSM8K(Cobbe et al. 2021), linguistically diverse grade school math word problems

    Examples:
        question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    """

    def __init__(self, args):
        self.name = "gsm8k"
        dataset = load_dataset('gsm8k', 'main')
        # dataset = load_from_disk("gsm8k")
        self.example_data = list(dataset["train"])
        self.evaluation_data = list(dataset["test"])

        self.answer_trigger = "\nTherefore, the answer (arabic numerals) is "
        super().__init__(args)
    def answer_cleansing(self, pred, must_choice=False):
        preds = pred.split(self.answer_trigger)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

        # If there is no candidate in list, null is set.
        if len(pred) == 0:
            pred = ""

        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]

        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]

        return pred

    def format_instance(self, instance):
        instance["answer"] = instance["answer"].replace("\n#### ", self.answer_trigger)
        instance["question"] = "Q: " + instance["question"] + "\n" + "A: "
        return dict(ground_truth=(instance["question"], instance["answer"]))


    @property
    def references(self):
        return [instance["answer"].split(self.answer_trigger)[-1] for instance in self.evaluation_data]
