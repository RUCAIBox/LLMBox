from .generation_dataset import GenerationDataset
from datasets import load_dataset, load_from_disk
from nltk.tokenize import word_tokenize
import numpy as np
import re
import string
import collections
import json


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

    def convert(self, raw_dataset, data_type):
        dataset = []
        for instance in raw_dataset:
            converted_instance = {}
            converted_instance["story"] = instance["story"]
            converted_instance["id"] = instance["id"]
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
            dataset.append(converted_instance)
        return dataset

    def format_instance(self, instance):
        num_questions = len(instance["questions"])
        return self._format_instances(instance, num_questions - 1)

    def normalize_answer(self, s):
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
    
    def get_tokens(self, s):
        if not s: return []
        return word_tokenize(self.normalize_answer(s))

    def compute_exact(self, reference, predicted):
        return int(self.normalize_answer(reference) == self.normalize_answer(predicted))

    def calculate_f1_score(self, reference, predicted):
        gold_toks = self.get_tokens(reference)
        pred_toks = self.get_tokens(predicted)
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

    def calculate_metric(self, predictions):
        f1_sum = 0
        em_sum = 0
        for prediction, reference in zip(predictions, self.references):
            f1_sum += max(self.calculate_f1_score(answer, prediction) for answer in reference)
            em_sum += max(self.compute_exact(answer, prediction) for answer in reference)
        return {'F1 score:': f1_sum / len(predictions),"em score": em_sum / len(predictions)}

    def construct_instances(self):
        self.evaluation_instances = []
        for instance in self.evaluation_data:
            formatted_instances = self.format_mul_instances(instance)
            self.evaluation_instances.extend(list(map(self.format_instruction_and_examples, formatted_instances)))

    def _format_instances(self, instance, qid):
        source_text = instance["story"]
        questions = [question + "?" if question[-1] != "?" else question for question in instance["questions"]]
        answers = instance["answers"]
        q_a_pair = "".join(
            map(lambda _p: "\n\nQ: " + _p[0] + "\n\n" + "A: " + _p[1][0], zip(questions[:qid], answers[:qid]))
        )
        source_text += q_a_pair
        source_text += "\n\nQ: " + questions[qid] + "\n\nA:"
        target_text = " " + answers[qid][0]
        return dict(source=source_text, target=target_text)

    def format_mul_instances(self, instance):
        mul_instances = []
        num_questions = len(instance["questions"])
        for i in range(num_questions):
            mul_instance = self._format_instances(instance, i)
            mul_instances.append(mul_instance["source"])
        return mul_instances

    @property
    def references(self):
        return [answer for instance in self.evaluation_data for answer in instance["answers"]]
