import numpy as np

from .multiple_choice_dataset import MultipleChoiceDataset


class Arc(MultipleChoiceDataset):
    """The dataset of ai2_arc.

        A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage
        research in advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where
        the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence
        algorithm. We are also including a corpus of over 14 million science sentences relevant to the task, and an
        implementation of three neural baseline models for this dataset. We pose ARC as a challenge to the community.

        Example:
            question:
            One year, the oak trees in a park began producing more acorns than usual. The next year,
            the population of chipmunks in the park also increased. Which best explains why there were more chipmunks
            the next year?

            answerKey: B

            choices: {
                label: [A, B, C, D],
                text: [Shady areas increased., Food sources increased., Oxygen levels increased., Available water increased.]
            }
        """

    instruction = ""
    evaluation_set = "test"
    example_set = "train"
    load_args = ("allenai/ai2_arc", "ARC-Challenge")

    def format_instance(self, instance):
        source_text = "Q: " + instance["question"] + "\n\nA:"
        options = instance["choices"]["text"]
        options = list(map(lambda _s: " " + _s, options))
        if instance["answerKey"].isdigit():
            target = options[ord(instance["answerKey"]) - 49]
        else:
            target = options[ord(instance["answerKey"]) - 65]
        return dict(
            source=source_text,
            target=target,
            options=options,
        )

    def construct_instances(self):
        self.evaluation_instances = []
        self.option_nums = []
        for instance in self.evaluation_data:
            formatted_instance = self.format_instance(instance)
            instance_with_examples = self.format_instruction_and_examples(formatted_instance)
            options = [(instance_with_examples, option) for option in formatted_instance['options']]
            self.option_nums.append(len(options))
            answer_options = [("A:", option) for option in formatted_instance["options"]]
            options = [item for pair in zip(options, answer_options) for item in pair]
            self.evaluation_instances.extend(options)
        print(self.option_nums)
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
        ref = []
        for instance in self.evaluation_data:
            if instance["answerKey"].isdigit():
                ref.append(ord(instance["answerKey"]) - 49)
            else:
                ref.append(ord(instance["answerKey"]) - 65)
        return ref

