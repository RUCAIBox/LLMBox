from .generation_dataset import GenerationDataset
from datasets import load_dataset, load_from_disk
import re
import evaluate


class Gsm8k(GenerationDataset):
    """The dataset of GSM8K.

    GSM8K(Cobbe et al. 2021), linguistically diverse grade school math word problems

    Examples:
        question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    """

    def __init__(self, args, model):
        self.name = "gsm8k"
        dataset = load_dataset('gsm8k', 'main')
        # dataset = load_from_disk("gsm8k")
        self.example_data = list(dataset["train"])
        self.evaluation_data = list(dataset["test"])
        self.instruction = "Answer the following questions."

        self.metric = "accuracy"
        self.answer_trigger = "\nTherefore, the answer (arabic numerals) is "
        super().__init__(args, model)

    def answer_cleansing(self, preds):
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
        predictions = self.answer_cleansing(predictions)
        exact_match = evaluate.load("exact_match")
        em_score = exact_match.compute(predictions=predictions, references=self.references, ignore_case=True, ignore_punctuation=True)["exact_match"]
        return {'Accuracy': em_score}

    @property
    def references(self):
        return [instance["answer"].split(self.answer_trigger)[-1] for instance in self.evaluation_data]
