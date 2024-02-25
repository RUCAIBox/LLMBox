from logging import getLogger

import numpy as np

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class Arc(MultipleChoiceDataset):
    """The dataset of ai2_arc.

        A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage
        research in advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where
        the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence
        algorithm.

        Example:
            'id': 'Mercury_7175875',
            'question': 'An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?',
            'choices': {
                'text': ['Planetary density will decrease.', 'Planetary years will become longer.', 'Planetary days will become shorter.', 'Planetary gravity will become stronger.'],
                'label': ['A', 'B', 'C', 'D']
            },
            'answerKey': 'C'
        """

    instruction = ""
    evaluation_set = "test"
    example_set = "train"
    load_args = ("allenai/ai2_arc",)

    def format_instance(self, instance):
        options = list(map(lambda _s: " " + _s, instance["choices"]["text"]))
        if instance["answerKey"].isdigit():
            instance["answerKey"] = ord(instance["answerKey"]) - 49
        else:
            instance["answerKey"] = ord(instance["answerKey"]) - 65
        return dict(
            source="Question: " + instance["question"],
            source_postfix="\nAnswer:",
            target_idx=instance["answerKey"],
            options=options,
        )

    def construct_instances(self):
        self.evaluation_instances = []
        self.option_nums = []
        for formatted_instance in self.formatted_evaluation_data:
            instance_with_examples = self.format_instruction_and_examples(formatted_instance)
            options = [(instance_with_examples, option) for option in formatted_instance['options']]
            self.option_nums.append(len(options))
            answer_options = [("Answer:", option) for option in formatted_instance["options"]]
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
        return [instance["answerKey"] for instance in self.evaluation_data]
