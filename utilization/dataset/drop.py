import re
from functools import cached_property

import datasets as ds

from ..metric import F1, Em
from ..metric.drop_utils import get_answers
from .generation_dataset import GenerationDataset


class Drop(GenerationDataset):
    """The dataset of Drop.

    DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs.

    Examples:
        passage: Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens.  The Texans would respond with fullback Vonta Leach getting a 1-yard touchdown run, yet the Raiders would answer with kicker Sebastian Janikowski getting a 33-yard and a 30-yard field goal.  Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal. Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt.
        question: Which team won the game
        answers:
           ['Raiders',
            'Raiders',
            'Raiders',
            'the Raiders',
            'the Raiders'],
    """

    instruction = "Answer the question based on the given passage.\n\nPassage: {passage}\nQuestion: {question}\nAnswer:"
    example_set = "train"
    evaluation_set = "validation"
    load_args = ("EleutherAI/drop",)
    load_kwargs = {"download_config": ds.DownloadConfig(extract_compressed_file=True)}
    metrics = [F1(force_number_match=True, word_tokenize="regex", align_bag="counter"), Em()]
    extra_model_args = dict(max_tokens=64, temperature=0, stop=["\n"])

    def format_instance(self, instance):
        if "spans" in instance:
            instance["target"] = instance["spans"][0]
        else:
            instance["target"] = instance["answers_spans"]["spans"][0]
        return instance

    @staticmethod
    def post_processing(predictions):
        new_predictions = []
        pattern = r"[.!(\n)]"
        for pred in predictions:
            match = re.search(pattern, pred)
            if match:
                index = match.start()
                pred = pred[:index]
            new_predictions.append(pred)
        return new_predictions

    @cached_property
    def references(self):
        return [instance["answers_spans"]["spans"] for instance in self.evaluation_data]
