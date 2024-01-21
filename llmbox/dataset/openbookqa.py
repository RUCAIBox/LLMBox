import numpy as np

from .multiple_choice_dataset import MultipleChoiceDataset


class OpenBookQA(MultipleChoiceDataset):
    """The dataset of WinoGrande.

    WinoGrande is a new collection of 44k problems, inspired by Winograd Schema Challenge
    (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and robustness against the
    dataset-specific bias. Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right
    option for a given sentence which requires commonsense reasoning.

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

    instruction = ""
    evaluation_set = "test"
    example_set = "train"
    load_args = ("openbookqa", "main")  # specify subset from command line

    def format_instance(self, instance):
        source_text = "Q: " + instance['question_stem'] + '\n' + 'A: '
        options = instance["choices"]['text']
        options = list(map(lambda _s: " " + _s, options))
        return dict(
            source=source_text,
            target=options[ord(instance["answerKey"]) - 65],
            options=options,
        )

    def construct_instances(self):
        self.evaluation_instances = []
        self.option_nums = []
        for instance in self.evaluation_data:
            formatted_instance = self.format_instance(instance)
            instance_with_examples = self.format_instruction_and_examples(formatted_instance)
            options = [(instance_with_examples, option) for option in formatted_instance['options']]
            # options = [
            #     self.format_instruction_and_examples(formatted_instance["source"], option)
            #     for option in formatted_instance["options"]
            # ]
            self.option_nums.append(len(options))
            answer_options = [("A:", option) for option in formatted_instance["options"]]
            options = [item for pair in zip(options, answer_options) for item in pair]
            self.evaluation_instances.extend(options)
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
