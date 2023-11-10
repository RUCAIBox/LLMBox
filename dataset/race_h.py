from .multiple_choice_dataset import MultipleChoiceDataset
from datasets import load_dataset, load_from_disk

class Race_h(MultipleChoiceDataset):
    """The dataset of RACE_h.

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

    def __init__(self, args):
        self.name = "race_h"
        dataset = load_dataset("race", "high")
        self.example_data = list(dataset[args.example_set])
        self.evaluation_data = list(dataset[args.evaluation_set])
        super().__init__(args)

    def format_instance(self, instance):
        source_text = "Article:\n" + instance["article"] + "\n\n" + "Q: " + instance["question"] + "\n\nA:"

        answer2text = {
            "A" : " " + instance["options"][0][0].lower() + instance["options"][0][1:],
            "B" : " " + instance["options"][1][0].lower() + instance["options"][1][1:],
            "C" : " " + instance["options"][2][0].lower() + instance["options"][2][1:],
            "D" : " " + instance["options"][3][0].lower() + instance["options"][3][1:],
        }

        options = []
        for option in ["A","B","C","D"]:
            target_text = answer2text[option]
            options.append((source_text, target_text))
        return dict(ground_truth=(source_text, answer2text[instance["answer"]]),options=options)

    @property
    def references(self):
        answer2index = {
            "A" : 0,
            "B" : 1,
            "C" : 2,
            "D" : 3,
        }
        return [answer2index[instance["answer"]] for instance in self.evaluation_data]
