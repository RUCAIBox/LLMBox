from functools import cached_property
from logging import getLogger

from ..metric import Accuracy
from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class OpenBookQA(MultipleChoiceDataset):
    """The dataset of OpenBookQA.

    OpenBookQA contains questions that require multi-step reasoning, use of additional common and commonsense knowledge, and rich text comprehension. OpenBookQA is a new kind of question-answering dataset modeled after open book exams for assessing human understanding of a subject.

    Example:
        'id': 8-343
        'question_stem': 'A person wants to start saving money so that they can afford a nice vacation at the end of the year. After looking over their budget and expenses, they decide the best way to save money is to'
        'choices': {
            'text': ['make more phone calls', 'quit eating lunch out', 'buy less with monopoly money', 'have lunch with friends']
            'label': ['A', 'B', 'C', 'D']
        }
        'answerKey': 'B'
    """

    instruction = "Q: {{question_stem}}{{'\n' + options if options}}\nAnswer:"
    evaluation_set = "test"
    example_set = "train"
    load_args = ("openbookqa", "main")
    use_normalization = True
    normalization_prompt = "Q: \nA:"
    metrics = [Accuracy("Norm")]

    def init_arguments(self):
        # TODO
        self.prefix_caching = False

    def format_instance(self, instance):
        instance["target_idx"] = ord(instance["answerKey"]) - 65
        instance["options"] = instance["choices"]['text']
        return instance

    @cached_property
    def references(self):
        return [ord(instance["answerKey"]) - 65 for instance in self.evaluation_data]
