from functools import cached_property

import re
from ..metric import F1, Em
from .generation_dataset import GenerationDataset


class Tydiqa(GenerationDataset):
    """The dataset of Tydiqa.

    TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs. The languages of TyDi QA are diverse with regard to their typology -- the set of linguistic features that each language expresses -- such that we expect models performing well on this set to generalize across a large number of the languages in the world. It contains language phenomena that would not be found in English-only corpora. To provide a realistic information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language without the use of translation (unlike MLQA and XQuAD).

    Examples:
        context: Charles Hoy Fort (6. elokuuta (joidenkin lähteiden mukaan 9.) 1874 – 3. toukokuuta 1932) oli yhdysvaltalainen kirjailija ja paranormaalien ilmiöiden tutkija.
        question:Milloin Charles Fort syntyi?
        answer: { "text": [ "6. elokuuta (joidenkin lähteiden mukaan 9.) 1874" ], "answer_start": [ 18 ] }
        title: Charles Fort
    """

    instruction = 'Answer each question using information in the preceding background paragraph.\n\nTitle: {title}\nBackground: {context}\n\nQ: {question}\n\nA:'
    example_set = "train"
    evaluation_set = "validation"
    load_args = ("tydiqa", "secondary_task")  # in order to support squad_v2, load_args is set in load.py
    metrics = [F1(), Em()]
    extra_model_args = dict(max_tokens=64, temperature=0, stop=["\n"])

    def format_instance(self, instance):
        instance["target"] = instance["answers"]["text"][0] 
        return instance

    @cached_property
    def references(self):
        return [instance["answers"]["text"][0] for instance in self.evaluation_data]

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
