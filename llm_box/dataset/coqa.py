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
        questions: When was the Vat formally opened?
        source: wikipedia
        story: The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, l..."
    """

    name = "coqa"
    instruction = "Answer the last question based the following story."

    example_set = "train"
    evaluation_set = "validation"

    load_args = ("coqa",)

    def _load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        evaluation_dataset = json.load(open("./coqa_data/dev.json"))["data"][:10]
        example_dataset = json.load(open("./coqa_data/train.json"))["data"]
        self.evaluation_data = Coqa.convert(evaluation_dataset, "dev")
        self.example_data = Coqa.convert(example_dataset, "train")

    @staticmethod
    def convert(raw_dataset, data_type):
        dataset = []
        for instance in raw_dataset:
            a = {}
            a["story"] = instance["story"]
            a["id"] = instance["id"]
            multiple_answers = [instance["answers"]]
            if data_type == "dev":
                multiple_answers += instance["additional_answers"].values()
            a["questions"] = []
            a["answers"] = []
            for i, qa in enumerate(instance["questions"]):
                a["questions"].append(qa["input_text"])
                answer_list = []
                for answers in multiple_answers:
                    answer = answers[i]
                    answer_list.append(answer["input_text"])
                a["answers"].append(answer_list)
            dataset.append(a)
        return dataset

    def format_instance(self, instance):
        num_questions = len(instance["questions"])
        return self._format_instances(instance, num_questions - 1)

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
    def get_tokens(s):
        if not s: return []
        return word_tokenize(Coqa.normalize_answer(s))

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(Coqa.normalize_answer(a_gold) == Coqa.normalize_answer(a_pred))

    @staticmethod
    def calculate_f1_score(reference, predicted):
        pattern = r'[\n]'  # 正则表达式模式，匹配`.`、`!` 或者换行符（`\n`）
        match = re.search(pattern, predicted)  # 在字符串中搜索第一个匹配项
        if match:
            index = match.start()
            predicted = predicted[:index]
        gold_toks = Coqa.get_tokens(reference)
        pred_toks = Coqa.get_tokens(predicted)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def calculate_metric(self, predictions):
        f1_sum = 0
        for prediction, reference in zip(predictions, self.references):
            f1_sum += max(Coqa.calculate_f1_score(answer, prediction) for answer in reference)
        # res_list = np.array(list(map(lambda _p: Drop.calculate_f1_score(_p[0],_p[1]), zip(self.references,predictions))))
        print(f1_sum)
        print(len(predictions))
        return {'F1 score: ': f1_sum / len(predictions)}

    def construct_instances(self):
        r"""Construct and format all the instances of `evaluation_data`.

        Returns:
            List[str]: The list of final formatted instances.
        """
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
