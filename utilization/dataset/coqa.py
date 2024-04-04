import copy
import json
import re
from pathlib import Path

from ..metric import F1, Em
from .generation_dataset import GenerationDataset


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

    instruction = "Answer the last question based on the given passage."
    example_set = "train"
    evaluation_set = "dev"
    load_args = ("coqa",)
    metrics = [F1(multiref_strategy="leave_one_out"), Em(multiref_strategy="leave_one_out")]
    extra_model_args = dict(max_tokens=64, temperature=0, stop=["\n"])

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        if not dataset_path:
            dataset_path = ""
        dataset_path = Path(dataset_path)

        try:
            # https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json
            evaluation_dataset = json.load(open(dataset_path / f"coqa-{self.evaluation_set}-v1.0.json"))["data"]
            # https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
            example_dataset = json.load(open(dataset_path / f"coqa-{self.example_set}-v1.0.json"))["data"]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"CoQA not found. Please download the dataset from https://stanfordnlp.github.io/coqa/coqa-dev-v1.0.json and https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json and specify the path to the dataset directory using the --dataset_path argument."
            )

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
                    dataset.append(copy.deepcopy(converted_instance))
            if data_type == "train":
                dataset.append((copy.deepcopy(converted_instance)))
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

    @property
    def references(self):
        return [instance["answers"][-1] for instance in self.evaluation_data]
