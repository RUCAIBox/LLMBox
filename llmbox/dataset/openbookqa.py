from logging import getLogger

import numpy as np

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

    instruction = ""
    evaluation_set = "test"
    example_set = "train"
    load_args = ("openbookqa", "main")

    def _format_instance(self, instance):
        source_text = "Q: " + instance['question_stem']
        options = instance["choices"]['text']
        options = list(map(lambda _s: " " + _s, options))
        return dict(
            source=source_text,
            source_postfix="\nA:",
            target_idx=ord(instance["answerKey"]) - 65,
            options=options,
        )

    def construct_instances(self):
        self.evaluation_instances = []
        self.option_nums = []
        for formatted_instance in self.formatted_evaluation_data:
            instance_with_examples = self.format_instruction_and_examples(formatted_instance)
            options = [(instance_with_examples, option) for option in formatted_instance['options']]
            self.option_nums.append(len(options))
            answer_options = [("A:", option) for option in formatted_instance["options"]]
            options = [item for pair in zip(options, answer_options) for item in pair]
            self.evaluation_instances.extend(options)
        logger.info("Evaluation mode: calculate PPL of the optional text based on the source text")
        logger.info("Formatted example (source)\n" + self.evaluation_instances[0][0])
        logger.info("Formatted example (option)\n" + self.evaluation_instances[0][1])
        self.evaluation_instances = self.evaluation_instances * self.args.sample_num

    def post_processing(self, predictions):
        labels = []
        st = 0
        predictions = list(map(lambda _r: _r[0], predictions))
        predictions = np.array([rc - ra for rc, ra in zip(predictions[::2], predictions[1::2])])
        for num in self.option_nums:
            labels.append(predictions[st:st + num].argmin())
            st += num
        predictions = labels
        return predictions

    @property
    def references(self):
        return [ord(instance["answerKey"]) - 65 for instance in self.evaluation_data]
