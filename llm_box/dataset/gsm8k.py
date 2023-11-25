import re

import numpy as np

from .generation_dataset import GenerationDataset


class Gsm8k(GenerationDataset):
    """The dataset of GSM8K.

    GSM8K(Cobbe et al. 2021), linguistically diverse grade school math word problems

    Examples:
        question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    """

    name = "gsm8k"
    instruction = "Answer the following question."
    answer_trigger = "\nTherefore, the answer (arabic numerals) is "

    evaluation_set = "test"
    example_set = "train"

    load_args = ("gsm8k", "main")

    @staticmethod
    def answer_cleansing(preds):
        predictions = []
        for pred in preds:
            # replace numbers like `x,xxx` with `xxxx`
            pred = re.sub(r"(\d),(\d)", r"\1\2", pred)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
            if numbers:
                predictions.append(numbers[-1])
            else:
                predictions.append(pred)

        return predictions

    def format_instance(self, instance):
        instance["answer"] = re.sub(r"(\d),(\d)", r"\1\2", instance["answer"])
        instance["answer"] = " " + instance["answer"].replace("\n#### ", self.answer_trigger)
        instance["question"] = "Q: " + instance["question"] + "\n" + "A:"
        return dict(
            source=instance["question"],
            target=instance["answer"],
        )

    def calculate_metric(self, predictions):
        score_list = np.asarray(predictions) == np.asarray(self.references)
        return {'Accuracy': np.mean(score_list)}

    @property
    def references(self):
        return [instance["answer"].split(self.answer_trigger)[-1] for instance in self.evaluation_data]
