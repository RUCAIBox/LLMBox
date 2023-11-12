from .multiple_choice_dataset import MultipleChoiceDataset
from datasets import load_dataset, load_from_disk
import numpy as np


class Race(MultipleChoiceDataset):
    """The dataset of RACE_h and RACE_m.

    The ReAding Comprehension dataset from Examinations (RACE) dataset is a machine reading comprehension dataset 
    consisting of 27,933 passages and 97,867 questions from English exams, targeting Chinese students aged 12-18.
    RACE consists of two subsets, RACE-M and RACE-H, from middle school and high school exams, respectively.
    RACE-M has 28,293 questions and RACE-H has 69,574.
    Each question is associated with 4 candidate answers, one of which is correct.

    Example:
        article: 
        The rain had continued for a week and the flood had created a big river which were ... with tears.

        question: What did Nancy try to do before she fell over?

        answer: C
        
        options': 
        [
        'Measure the depth of the river',
        'Look for a fallen tree trunk',
        'Protect her cows from being drowned',
        'Run away from the flooded farm'
        ]
    """

    def __init__(self, args, model):
        self.name = "race_" + args.dataset[1]
        if args.dataset[1] in ["h","high"]:
            dataset = load_dataset("race", "high")
            # dataset = load_from_disk("../dataset/race_h")
        elif args.dataset[1] in ["m","middle"]:
            dataset = load_dataset("race", "middle")
            # dataset = load_from_disk("../dataset/race_m")
        self.example_data = list(dataset[args.example_set])
        self.evaluation_data = list(dataset[args.evaluation_set])
        super().__init__(args, model)

    def format_instance(self, instance):
        source_text = "Article:\n" + instance["article"] + "\n\n" + "Q: " + instance["question"] + "\n\nA:"
        options = instance["options"]
        options = list(map(lambda _s: " " + _s, options))
        return dict(
            source=source_text,
            target=options[ord(instance["answer"]) - 65],
            options=options,
        )

    def construct_instances(self):
        self.evaluation_instances = []
        self.option_nums = []
        for instance in self.evaluation_data:
            formatted_instance = self.format_instance(instance)
            options = [
                self.format_instruction_and_examples(formatted_instance["source"], option)
                for option in formatted_instance["options"]
            ]
            self.option_nums.append(len(options))
            answer_options = [("A:", option) for option in formatted_instance["options"]]
            options = [item for pair in zip(options, answer_options) for item in pair]
            self.evaluation_instances.extend(options)

    def calculate_metric(self, results):
        labels = []
        st = 0
        results = list(map(lambda _r: _r[0], results))
        results = np.array([rc - ra for rc, ra in zip(results[::2], results[1::2])])
        for num in self.option_nums:
            labels.append(results[st:st + num].argmin())
            st += num
        results = labels
        score_list = np.asarray(results) == np.asarray(self.references)
        return {'Accuracy': np.mean(score_list)}

    @property
    def references(self):
        return [ord(instance["answer"]) - 65 for instance in self.evaluation_data]
