import re

from ..metric import F1, Em
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

    instruction = "Answer the question based on the given passage."
    example_set = "train"
    evaluation_set = "validation"
    load_args = ("drop",)
    metrics = [F1(force_number_match=True), Em()]
    model_args = dict(max_tokens=64, temperature=0, stop=['\n'])

    def format_instance(self, instance):
        source_text = "Passage: " + instance["passage"] + "\nQuestion: " + instance["question"] + "\nAnswer:"
        target_text = " " + instance["answers_spans"]["spans"][0]
        return dict(source=source_text, target=target_text)

    @staticmethod
    def post_processing(predictions):
        new_predictions = []
        pattern = r'[.!(\n)]'
        for pred in predictions:
            match = re.search(pattern, pred)
            if match:
                index = match.start()
                pred = pred[:index]
            new_predictions.append(pred)
        return new_predictions

    @property
    def references(self):
        return [instance["answers_spans"]["spans"] for instance in self.evaluation_data]
