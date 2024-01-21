import random
import re

import numpy as np

from ..metric import F1, Em
from .generation_dataset import GenerationDataset


class Squad_v2(GenerationDataset):
    """The dataset of Squad_v2.

    Gcombines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones.

    Examples:
        context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
        question: In what country is Normandy located?
        answer: ['France', 'France', 'France', 'France']
    """

    instruction = 'Answer each question using information in the preceding background paragraph.\nIf there is not enough information provided, answer with "Not in background."'
    example_set = "train"
    evaluation_set = "validation"
    load_args = ("squad_v2",)
    metrics = [F1(), Em()]
    extra_model_args = dict(max_tokens=64, temperature=0, stop=["\n"])

    def format_instance(self, instance):
        source_text = (
            "Title: " + instance["title"] + "\n\nBackground: " + instance["context"] + "\n\nQ: " +
            instance["question"] + "\n\nA:"
        )
        text = instance["answers"]["text"]
        if not text:
            text = "Not in background."
        else:
            text = text[0]
        target_text = " " + text
        return dict(source=source_text, target=target_text)

    def construct_examples(self, instance=None) -> str:
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Dict): a pre-formatted evaluation instance.

        Returns:
            str: The constructed demonstration text.
        """
        if self.num_shots == 0:
            return ""
        elif len(self.example_data) == 0:
            raise ValueError(
                f"Receive num_shots={self.num_shots}, but cannot construct examples for dataset {self.name} without example data."
            )

        generation_example_text = ""
        generation_example_token_nums = 0
        classified_title = {}
        for item in self.example_data:
            key = (item["title"], item["context"])
            if key in classified_title:
                classified_title[key].append(item)
            else:
                classified_title[key] = [item]

        keys = list(classified_title.keys())
        randoms_keys = [keys[i] for i in np.random.choice(range(len(keys)), self.num_shots, replace=False)]
        for data in [classified_title[key] for key in randoms_keys]:
            instance = data[0]
            source_text = "Title: " + instance["title"] + "\n\nBackground: " + instance["context"]
            random.shuffle(data)
            for instance in data:
                source_text += "\n\nQ: " + instance["question"] + "\n\nA:"
                text = instance["answers"]["text"]
                if not text:
                    text = "Not in background."
                else:
                    text = text[0]
                target_text = " " + text
                source_text += target_text
            cur_example_text = source_text + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + generation_example_token_nums <= self.max_example_tokens:
                generation_example_text += cur_example_text
                generation_example_token_nums += cur_token_num
        return generation_example_text

    @property
    def references(self):
        return [
            instance["answers"]["text"] if instance["answers"]["text"] else ["Not in background."]
            for instance in self.evaluation_data
        ]

    def post_processing(self, preds):
        predictions = []
        pattern = r"[.!(\n)]"
        for pred in preds:
            match = re.search(pattern, pred)
            if match:
                index = match.start()
                pred = pred[:index]
            predictions.append(pred)
        return predictions
