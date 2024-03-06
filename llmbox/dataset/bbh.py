from logging import getLogger

from .generation_dataset import GenerationDataset
from .enum import BBH_PROMPTS, BBH_NO_CHOICE, BBH_LETTER_CHOICE
from ..metric import Em

logger = getLogger(__name__)

import re

class Bbh(GenerationDataset):
    """The dataset of BBH.

    Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them by Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason.

    Example:
        "input": "Question: Fidel tells the truth. Jerry says Fidel tells the truth. Vina says Jerry tells the truth. Millicent says Vina lies. Raymond says Millicent lies. Does Raymond tell the truth?",
        "answer": "Let's think step by step. (1) Fidel tells the truth. So, we know that Fidel tells the truth. (2) Jerry says Fidel tells the truth. Since we know from (1) that Fidel tells the truth, if Jerry says that Fidel tells the truth, then Jerry tells the truth. (3) Vina says Jerry tells the truth. Since we know from (2) that Jerry tells the truth, if Vina says Jerry tells the truth, then Vine tells the truth. (4) Millicent says Vina lies. Since we know from (3) that Vina tells the truth, if Millicent says Vina lies, then Millicent lies. (5) Raymond says Millicent lies. Since we know from (4) that Millicent lies, if Raymond says Millicent lies, then Raymond tells the truth. Now, the question asks: Does Raymond tell the truth? We know from (5) that Raymond tells the truth. So the answer is Yes.",
        "label": "Yes"
    """

    instruction = "{}"
    example_set = "dev"
    evaluation_set = "test"
    load_args = ("RUCAIBox/bbh",)
    metrics = [Em()]

    def __init__(self, args, model, subset_name: str):
        self.instruction = self.instruction.format(BBH_PROMPTS[subset_name])
        self.task = subset_name
        self.extra_model_args = dict(stop=["\n"]) if args.cot is None else dict()
        super().__init__(args, model, subset_name)

    def format_instance(self, instance):
        target = instance["answer"]
        if target is None or self.args.cot is None:
            target = instance["label"]
        return dict(
            source="Q: " + instance["input"].strip() + "\nA:",
            target=" " + target
        )

    def post_processing(self, predictions):
        new_predictions = []
        for pred in predictions:
            extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", pred)
            if extracted_answer:
                new_pred = extracted_answer.group(1).strip()
            else:
                new_pred = pred.strip()
            if self.task not in BBH_NO_CHOICE:
                if len(new_pred) > 0:
                    new_pred = new_pred.split()[0]
                    if self.task in BBH_LETTER_CHOICE:
                        pattern = r'.*([A-Z]).*'
                        new_pred = re.sub(pattern, r'\1', new_pred)
            new_predictions.append(new_pred)
        return new_predictions

    @property
    def references(self):
        references = []
        for instance in self.evaluation_data:
            label = instance["label"]
            pattern = r'\(([A-Z])\)'
            label = re.sub(pattern, r'\1', label)
            references.append([label])
        return references
