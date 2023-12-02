from .generation_dataset import GenerationDataset
import numpy as np
import re
import string
import collections
from nltk import word_tokenize


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

    name = "drop"
    instruction = "Answer the question based on the given passage."

    example_set = "train"
    evaluation_set = "validation"

    load_args = ("drop",)

    def format_instance(self, instance):
        source_text = "Passage: " + instance["passage"] + "\nQuestion: " + instance["question"] + "\nAnswer:"
        target_text = " " + instance["answers_spans"]["spans"][0]
        return dict(source=source_text, target=target_text)

    def calculate_metric(self, predictions):
        f1_sum = 0
        em_sum = 0
        for prediction, references in zip(predictions, self.references):
            f1_sum += self._metric_max_over_ground_truths(self.calculate_f1_score, prediction, references)
            em_sum += self._metric_max_over_ground_truths(self.calculate_exact_match_score, prediction, references)
        return {'F1': f1_sum / len(predictions) * 100, "EM": em_sum / len(predictions) * 100}

    @property
    def references(self):
        return [instance["answers_spans"]["spans"] for instance in self.evaluation_data]

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, stories and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def calculate_exact_match_score(reference, predicted):
        return int(Drop.normalize_answer(reference) == Drop.normalize_answer(predicted))

    @staticmethod
    def calculate_f1_score(reference, predicted):
        gold_toks = word_tokenize(Drop.normalize_answer(reference))
        pred_toks = word_tokenize(Drop.normalize_answer(predicted))
        contain_num, gold_numbers, predicted_numbers  = Drop._match_numbers_if_present(gold_toks, pred_toks)
        if contain_num:
            return 1 if gold_numbers.intersection(predicted_numbers) else 0
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def _match_numbers_if_present(gold_bag, predicted_bag):
        gold_numbers = set()
        predicted_numbers = set()
        contain_num = False
        for word in gold_bag:
            if Drop._is_number(word):
                gold_numbers.add(word)
        for word in predicted_bag:
            if Drop._is_number(word):
                predicted_numbers.add(word)
        if gold_numbers:
            contain_num = True
        return (contain_num, gold_numbers, predicted_numbers)

    @staticmethod
    def _is_number(text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def post_processing(preds):
        predictions = []
        pattern = r'[.!(\n)]'
        for pred in preds:
            match = re.search(pattern, pred)
            if match:
                index = match.start()
                pred = pred[:index]
            predictions.append(pred)
        return predictions

    @staticmethod
    def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
