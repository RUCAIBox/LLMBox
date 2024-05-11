from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


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

    instruction = "Article:\n{{article}}\n\nQ: {{question}}{{'\n' + options if options}}\nA:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("ehovy/race",)  # specify subset from command line, remove "all" by default
    banned_subsets = ["all"]
    use_normalization = True
    normalization_prompt = "Article:\n\nQ: \nA:"

    def init_arguments(self):
        # TODO
        self.prefix_caching = False

    def format_instance(self, instance):
        instance["target_idx"] = ord(instance["answer"]) - 65
        return instance

    @cached_property
    def references(self):
        return [ord(instance["answer"]) - 65 for instance in self.evaluation_data]
