from logging import getLogger

import numpy as np

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class WinoGrande(MultipleChoiceDataset):
    """The dataset of WinoGrande.

    WinoGrande is a new collection of 44k problems, inspired by Winograd Schema Challenge
    (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and robustness against the
    dataset-specific bias. Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right
    option for a given sentence which requires commonsense reasoning.

    Example:
        'answer': '2',
        'option1': 'Sarah',
        'option2': 'Maria',
        'sentence': 'Sarah was a much better surgeon than Maria so _ always got the easier cases.'
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("winogrande", "winogrande_debiased")  # specify subset from command line

    def format_instance(self, instance):
        text = instance['sentence'].split('_')
        source_text = [text[0] + instance[option] for option in ['option1', 'option2']]
        options = [text[1]] * 2
        return dict(
            source=source_text,
            target=source_text[int(instance["answer"]) - 1],
            options=options,
        )

    @property
    def references(self):
        return [int(instance["answer"]) - 1 for instance in self.evaluation_data]

    def construct_examples(self, instance=None) -> str:
        if self.num_shots == 0:
            return ""
        indice = self.random_indice
        example_text = ""
        example_token_nums = 0
        for index in indice:
            if hasattr(self, "formatted_example_data"):
                example = self.formatted_example_data[index]
            else:
                example = self.format_instance(self.example_data[index])
            cur_example_text = self.args.instance_format.format(
                source=example["target"], target=example["options"][0]
            ) + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + example_token_nums <= self.max_example_tokens:
                example_text += cur_example_text
                example_token_nums += cur_token_num

        return example_text

    def construct_instances(self):
        self.evaluation_instances = []
        self.option_nums = []
        for formatted_instance in self.formatted_evaluation_data:
            for source, option in zip(formatted_instance['source'], formatted_instance['options']):
                if self.examples == "":
                    self.examples = self.construct_examples()
                if self.model.type == "base":
                    source = self.examples + self.args.instance_format.format(source=source, target="")
                elif self.model.type == "instruction":
                    source = (
                        self.instruction + "\n\n" + self.examples +
                        self.args.instance_format.format(source=source, target="")
                    )
                self.evaluation_instances.append((source, option))
            self.option_nums.append(2)

        logger.info("Evaluation mode: calculate PPL of the optional text based on the source text")
        logger.info("Formatted example (source)\n" + self.evaluation_instances[0][0])
        logger.info("Formatted example (option)\n" + self.evaluation_instances[0][1])
