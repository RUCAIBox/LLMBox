import re
from functools import cached_property
from logging import getLogger
from typing import List

from ..dataset_enum import BBH_LETTER_CHOICE, BBH_NO_CHOICE
from ..metric import Em
from .generation_dataset import GenerationDataset

logger = getLogger(__name__)

BBH_PROMPTS = {
    'boolean_expressions': "Evaluate the result of a random Boolean expression.",
    'causal_judgement': "Answer questions about causal attribution.",
    'date_understanding': "Infer the date from context.",
    'disambiguation_qa': "Clarify the meaning of sentences with ambiguous pronouns.",
    'dyck_languages': "Correctly close a Dyck-n word.",
    'formal_fallacies': "Distinguish deductively valid arguments from formal fallacies.",
    'geometric_shapes': "Name geometric shapes from their SVG paths.",
    'hyperbaton': "Order adjectives correctly in English sentences.",
    'logical_deduction_five_objects':
    "A logical deduction task which requires deducing the order of a sequence of objects.",
    'logical_deduction_seven_objects':
    "A logical deduction task which requires deducing the order of a sequence of objects.",
    'logical_deduction_three_objects':
    "A logical deduction task which requires deducing the order of a sequence of objects.",
    'movie_recommendation': "Recommend movies similar to the given list of movies.",
    'multistep_arithmetic_two': "Solve multi-step arithmetic problems.",
    'navigate':
    "Given a series of navigation instructions, determine whether one would end up back at the starting point.",
    'object_counting': "Questions that involve enumerating objects and asking the model to count them.",
    'penguins_in_a_table': "Answer questions about a table of penguins and their attributes.",
    'reasoning_about_colored_objects': "Answer extremely simple questions about the colors of objects on a surface.",
    'ruin_names': "Select the humorous edit that 'ruins' the input movie or musical artist name.",
    'salient_translation_error_detection':
    "Detect the type of error in an English translation of a German source sentence.",
    'snarks': "Determine which of two sentences is sarcastic.",
    'sports_understanding':
    "Determine whether an artificially constructed sentence relating to sports is plausible or not.",
    'temporal_sequences': "Task description: Answer questions about which times certain events could have occurred.",
    'tracking_shuffled_objects_five_objects':
    "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    'tracking_shuffled_objects_seven_objects':
    "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    'tracking_shuffled_objects_three_objects':
    "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    'web_of_lies': "Evaluate a random boolean function expressed as a word problem.",
    'word_sorting': "Sort a list of words."
}


class Bbh(GenerationDataset):
    """The dataset of BBH.

    Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them by Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason.

    Example:
        "input": "Question: Fidel tells the truth. Jerry says Fidel tells the truth. Vina says Jerry tells the truth. Millicent says Vina lies. Raymond says Millicent lies. Does Raymond tell the truth?",
        "answer": "Let's think step by step. (1) Fidel tells the truth. So, we know that Fidel tells the truth. (2) Jerry says Fidel tells the truth. Since we know from (1) that Fidel tells the truth, if Jerry says that Fidel tells the truth, then Jerry tells the truth. (3) Vina says Jerry tells the truth. Since we know from (2) that Jerry tells the truth, if Vina says Jerry tells the truth, then Vine tells the truth. (4) Millicent says Vina lies. Since we know from (3) that Vina tells the truth, if Millicent says Vina lies, then Millicent lies. (5) Raymond says Millicent lies. Since we know from (4) that Millicent lies, if Raymond says Millicent lies, then Raymond tells the truth. Now, the question asks: Does Raymond tell the truth? We know from (5) that Raymond tells the truth. So the answer is Yes.",
        "label": "Yes"
    """

    instruction = "{bbh_instruction}\n\nQ: {input}\nA:"
    example_set = "dev"
    evaluation_set = "test"
    load_args = ("RUCAIBox/bbh",)
    metrics = [Em()]
    supported_cot = ["base"]

    def init_arguments(self):
        self.bbh_instruction = BBH_PROMPTS[self.subset_name]
        if self.cot is None:
            # when using chain-of-thought, responses might be in multiple lines
            self.extra_model_args["stop"] = ["\n"]

    def format_instance(self, instance):
        target = instance["answer"]
        if target is None or self.cot is None:
            target = instance["label"]
        return dict(input=instance["input"].strip(), target=" " + target, bbh_instruction=self.bbh_instruction)

    def post_processing(self, predictions: List[str]):
        new_predictions = []
        for pred in predictions:
            extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", pred)
            if extracted_answer:
                new_pred = extracted_answer.group(1).strip()
            else:
                new_pred = pred.strip()
            if self.subset_name not in BBH_NO_CHOICE:
                if len(new_pred) > 0:
                    new_pred = new_pred.split()[0]
                    if self.subset_name in BBH_LETTER_CHOICE:
                        pattern = r'.*([A-Z]).*'
                        new_pred = re.sub(pattern, r'\1', new_pred)
            new_predictions.append(new_pred)
        return new_predictions

    @cached_property
    def references(self):
        references = []
        for instance in self.evaluation_data:
            label = instance["label"]
            pattern = r'\(([A-Z])\)'
            label = re.sub(pattern, r'\1', label)
            references.append([label])
        return references
