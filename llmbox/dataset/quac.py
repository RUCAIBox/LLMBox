import re

import numpy as np

from ..metric import F1, Em
from .generation_dataset import GenerationDataset


class Quac(GenerationDataset):
    """The dataset of quac.

    Question Answering in Context is a dataset for modeling, understanding, and participating in information seeking dialog.

    Examples:
        section_title: Chattanooga and Birmingham: 1926-29
        context: A former friend from the Mobile slums, Alex Herman, was the player/manager for the Chattanooga White Sox of the minor Negro Southern League. In 1926 he discovered Paige and offered to pay him $250 per month, of which Paige would collect $50 with the rest going to his mother. He also agreed to pay Lula Paige a $200 advance, and she agreed to the contract. The local newspapers--the Chattanooga News and Chattanooga Times--recognized from the beginning that Paige was special. In April 1926, shortly after his arrival, he recorded nine strikeouts over six innings against the Atlanta Black Crackers. Part way through the 1927 season, Paige\'s contract was sold to the Birmingham Black Barons of the major Negro National League (NNL). According to Paige\'s first memoir, his contract was for $450 per month, but in his second he said it was for $275. Pitching for the Black Barons, Paige threw hard but was wild and awkward. In his first big game in late June 1927, against the St. Louis Stars, Paige incited a brawl when his fastball hit the hand of St. Louis catcher Mitchell Murray. Murray then charged the mound and Paige raced for the dugout, but Murray flung his bat and struck Paige above the hip. The police were summoned, and the headline of the Birmingham Reporter proclaimed a "Near Riot." Paige improved and matured as a pitcher with help from his teammates, Sam Streeter and Harry Salmon, and his manager, Bill Gatewood. He finished the 1927 season 7-1 with 69 strikeouts and 26 walks in 89 1/3 innings. Over the next two seasons, Paige went 12-5 and 10-9 while recording 176 strikeouts in 1929. (Several sources credit his 1929 strikeout total as the all-time single-season record for the Negro leagues, though there is variation among the sources about the exact number of strikeouts.) On April 29 of that season he recorded 17 strikeouts in a game against the Cuban Stars, which exceeded what was then the major league record of 16 held by Noodles Hahn and Rube Waddell. Six days later he struck out 18 Nashville Elite Giants, a number that was tied in the white majors by Bob Feller in 1938. Due to his increased earning potential, Barons owner R. T. Jackson would "rent" Paige out to other ball clubs for a game or two to draw a decent crowd, with both Jackson and Paige taking a cut.
        questions:
               ['what did he do in Chattanooga',
                'how did he discover him',
                'what position did he play',
                'how did they help him',
                'when did he go to Birmingham',
                'how did he feel about this',
                'how did he do with this team',
                'What made him leave the team']
        answers:
               ['Alex Herman, was the player/manager for the Chattanooga White Sox of the minor Negro Southern League. In 1926 he discovered Paige and offered to pay him $250 per month,',
                'CANNOTANSWER',
                'Paige improved and matured as a pitcher with help from his teammates,',
                'CANNOTANSWER',
                "Part way through the 1927 season, Paige's contract was sold to the Birmingham Black Barons",
                'CANNOTANSWER',
                'Over the next two seasons, Paige went 12-5 and 10-9 while recording 176 strikeouts in 1929. (',
                'Jackson would "rent" Paige out to other ball clubs for a game or two to draw a decent crowd, with both Jackson and Paige taking a cut.']
    """

    instruction = """Answer each question using information in the preceding background paragraph. If there is not enough information provided, answer with "I don't know." """
    example_set = "train"
    evaluation_set = "validation"
    load_args = ("quac",)
    metrics = [F1(), Em()]
    extra_model_args = dict(max_tokens=64, temperature=0, stop=["\n"])

    def load_raw_dataset(
        self, dataset_path: str | None, subset_name: str | None, evaluation_set: str, example_set: str | None
    ):
        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, example_set)
        _evaluation_data = []
        for data in self.evaluation_data:
            questions = data["questions"]
            answers = data["answers"]["texts"]
            for question, answer in zip(questions, answers):
                formatted_data = {}
                formatted_data["question"] = question
                formatted_data["answer"] = answer
                formatted_data["title"] = data["section_title"]
                formatted_data["paragraph"] = data["context"]
                _evaluation_data.append(formatted_data)
        self.evaluation_data = _evaluation_data

    def format_instance(self, instance):
        source_text = (
            "TITLE: " + instance["title"] + "\nPARAGRAPH: " + instance["paragraph"] + "\n\nQ: " + instance["question"] +
            "\n\nA:"
        )
        text = instance["answer"]
        if "CANNOTANSWER" in text:
            text = "I don't know."
        else:
            text = text[0]
        target_text = " " + text
        return dict(source=source_text, target=target_text)

    def construct_examples(self, instance=None) -> str:
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Dict): a pre-formatted evaluation instance.

        Returns:
            str: The constructed demonstration text.
        """
        if self.num_shots == 0:
            return ""
        elif len(self.example_data) == 0:
            raise ValueError(
                f"Receive num_shots={self.num_shots}, but cannot construct examples for dataset {self.name} without example data."
            )

        example_text = ""
        example_token_nums = 0
        randoms_indices = np.random.choice(len(self.example_data), self.num_shots)
        for index in randoms_indices:
            instance = self.example_data[index]
            source_text = "TITLE: " + instance["section_title"] + "\n\nPARAGRAPH: " + instance["background"]
            for question, answers in zip(instance["questions"], instance["answers"]["texts"]):
                source_text += "\n\nQ: " + question + "\n\nA:"
                text = answers[0]
                if "CANNOTANSWER" in answers:
                    text = "I don't know."
                target_text = " " + text
                source_text += target_text
            cur_example_text = source_text + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + example_token_nums <= self.max_example_tokens:
                example_text += cur_example_text
                example_token_nums += cur_token_num
        return example_text

    @property
    def references(self):
        return [["I don't know."] if "CANNOTANSWER" in instance["answer"] else instance["answer"]
                for instance in self.evaluation_data]

    @staticmethod
    def post_processing(preds):
        predictions = []
        pattern = r"[.!(\n)]"
        for pred in preds:
            match = re.search(pattern, pred)
            if match:
                index = match.start()
                pred = pred[:index]
            predictions.append(pred)
        return predictions
