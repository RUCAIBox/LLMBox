from .multiple_choice_dataset import MultipleChoiceDataset
from datasets import load_dataset, load_from_disk
import random


class Race_m(MultipleChoiceDataset):
    """The dataset of RACE_m.

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
        self.name = "race_m"
        dataset = load_dataset("race", "middle")
        # dataset = load_from_disk("../dataset/race_m")
        self.example_data = list(dataset[args.example_set])
        raw_data = list(dataset[args.evaluation_set])
        self.evaluation_data = random.sample(raw_data, 500)
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

    @property
    def references(self):
        return [ord(instance["answer"]) - 65 for instance in self.evaluation_data]
