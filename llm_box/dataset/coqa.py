from .generation_dataset import GenerationDataset
import numpy as np
import re
import string
import collections
import json
from nltk import word_tokenize


class Coqa(GenerationDataset):
    """The dataset of CoQA.

    CoQA is a large-scale dataset for building Conversational Question Answering systems.

    Examples:
        answers:
               [['white', 'white', 'white', 'white'],
                ['in a barn', 'in a barn', 'in a barn', 'in a barn near'],
                ['no', 'no', 'No', 'no'],
                ['with her mommy and 5 sisters', 'her mommy and 5 other sisters', 'her mommy and 5 other sisters', 'her mommy and 5 other sisters'],
                ['orange and white', 'orange with white tiger stripes', 'orange', 'orange'],
                ['no', 'no', 'No', 'no'],
                ['she painted herself', 'she painted herself', 'paint herself like them', 'paint herself like them'],
                ['the farmer', "the farmer's", "the old farmer's", "the farmer's"],
                ['they started laughing', 'they started laughing', 'rubbed her face', 'started laughing'],
                ['a bucket of water', 'dropped her into a big bucket of water', 'into a big bucket of water', 'a big bucket of water'],
                ['licked her face', 'licked her face', 'licked her face', 'licked her face'],
                ['no', 'no', 'No', 'no']]
        questions: 
               ['What color was Cotton?',
                'Where did she live?',
                'Did she live alone?',
                'Who did she live with?',
                'What color were her sisters?',
                'Was Cotton happy that she looked different than the rest of her family?',
                'What did she do to try to make herself the same color as her sisters?',
                'Whose paint was it?',
                "What did Cotton's mother and siblings do when they saw her painted orange?",
                "Where did Cotton's mother put her to clean the paint off?",
                'What did the other cats do when Cotton emerged from the bucket of water?',
                'Did they want Cotton to change the color of her fur?']
        source: 'mctest',
        story:  'Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton. Cotton lived high up in a nice warm place above the barn where all of the farmer\'s horses slept. But Cotton wasn\'t alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters. All of her sisters were cute and fluffy, like Cotton. But she was the only white one in the bunch. The rest of her sisters were all orange with beautiful white tiger stripes like Cotton\'s mommy. Being different made Cotton quite sad. She often wished she looked like the rest of her family. So one day, when Cotton found a can of the old farmer\'s orange paint, she used it to paint herself like them. When her mommy and sisters found her they started laughing. \n\n"What are you doing, Cotton?!" \n\n"I only wanted to be more like you". \n\nCotton\'s mommy rubbed her face on Cotton\'s and said "Oh Cotton, but your fur is so pretty and special, like you. We would never want you to be any other way". And with that, Cotton\'s mommy picked her up and dropped her into a big bucket of water. When Cotton came out she was herself again. Her sisters licked her face until Cotton\'s fur was all all dry. \n\n"Don\'t ever do that again, Cotton!" they all cried. "Next time you might mess up that pretty white fur of yours and we wouldn\'t want that!" \n\nThen Cotton thought, "I change my mind. I like being special".' 
    """

    name = "coqa"
    instruction = "Answer the last question based on the given passage."

    example_set = "train"
    evaluation_set = "validation"

    load_args = ("coqa",)

    def _load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        evaluation_dataset = json.load(open("./llm_box/dataset/coqa_data/dev.json"))["data"]
        example_dataset = json.load(open("./llm_box/dataset/coqa_data/train.json"))["data"]
        self.evaluation_data = self.convert(evaluation_dataset, "dev")
        self.example_data = self.convert(example_dataset, "train")

    @staticmethod
    def convert(raw_dataset, data_type):
        dataset = []
        for instance in raw_dataset:
            converted_instance = {}
            converted_instance["story"] = instance["story"]
            multiple_answers = [instance["answers"]]
            if data_type == "dev":
                multiple_answers += instance["additional_answers"].values()
            converted_instance["questions"] = []
            converted_instance["answers"] = []
            for i, qa in enumerate(instance["questions"]):
                converted_instance["questions"].append(qa["input_text"])
                answer_list = []
                for answers in multiple_answers:
                    answer = answers[i]
                    answer_list.append(answer["input_text"])
                converted_instance["answers"].append(answer_list)
                if data_type == "dev":
                    dataset.append(converted_instance)
            if data_type == "train":
                dataset.append(converted_instance)
        return dataset

    def format_instance(self, instance):
        source_text = instance["story"]
        questions = [question + "?" if question[-1] != "?" else question for question in instance["questions"]]
        answers = instance["answers"]
        q_a_pair = "".join(
            map(lambda _p: "\n\nQ: " + _p[0] + "\n\n" + "A: " + _p[1][0], zip(questions[:-1], answers[:-1]))
        )
        source_text += q_a_pair
        source_text += "\n\nQ: " + questions[-1] + "\n\nA:"
        target_text = " " + answers[-1][0]
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
        return [instance["answers"][-1] for instance in self.evaluation_data]

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
        return int(Coqa.normalize_answer(reference) == Coqa.normalize_answer(predicted))

    @staticmethod
    def calculate_f1_score(reference, predicted):
        gold_toks = word_tokenize(Coqa.normalize_answer(reference))
        pred_toks = word_tokenize(Coqa.normalize_answer(predicted))
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

    @staticmethod
    def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        if len(scores_for_ground_truths) > 1:
            func = lambda x: (max(x) * (len(x) - 1) + np.partition(x, -2)[-2]) / len(x)
        else:
            func = max
        return func(scores_for_ground_truths)