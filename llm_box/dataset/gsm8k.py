import re

import numpy as np

from .generation_dataset import GenerationDataset
from ..metric import Accuracy


class Gsm8k(GenerationDataset):
    """The dataset of GSM8K.

    GSM8K(Cobbe et al. 2021), linguistically diverse grade school math word problems

    Examples:
        question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    """

    name = "gsm8k"
    instruction = "Answer the following question."

    evaluation_set = "test"
    example_set = "train"

    load_args = ("gsm8k", "main")
    metrics = [Accuracy()]

    model_args = dict(stop_sequences=["\n"], do_sample=False)
    # GSM8K extracts the last number in the predictions as the answer, so it's important to set a stop sequence for non-chat format

    answer_trigger = "Therefore, the answer (arabic numerals) is"
    question_pattern = "Question: {question}\nAnswer:"
    one_line_answer = True

    @staticmethod
    def post_processing(predictions):
        new_predictions = []
        for pred in predictions:
            # replace numbers like `x,xxx` with `xxxx`
            pred = re.sub(r"(\d),(\d)", r"\1\2", pred)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
            if numbers:
                new_predictions.append(numbers[-1])
            else:
                new_predictions.append(pred)
        return new_predictions

    def format_instance(self, instance):
        instance["answer"] = re.sub(r"(\d),(\d)", r"\1\2", instance["answer"])
        # instance["answer"] = re.sub(r"<<[\d\+\(\)-=\*\/]+>>", "", instance["answer"])
        # instance["question"] = re.sub(r"<<[\d\+\(\)-=\*\/]+>>", "", instance["question"])

        if self.one_line_answer:
            instance["answer"] = " " + instance["answer"].replace("\n#### ", " " + self.answer_trigger.strip() + " ")
            instance["answer"] = instance["answer"].replace("\n", " ")
            instance["question"] = instance["question"].replace("\n", " ")
        else:
            instance["answer"] = " " + instance["answer"].replace("\n#### ", "\n" + self.answer_trigger.strip() + " ")

        instance["question"] = self.question_pattern.format(question=instance["question"])
        return dict(
            source=instance["question"],
            target=instance["answer"],
        )

    @property
    def references(self):
        if not hasattr(self, "_references"):
            self._references = [
                instance["answer"].split(self.answer_trigger)[-1].strip() for instance in self.evaluation_data
            ]
        return self._references
