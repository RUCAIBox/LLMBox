from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Qqp(MultipleChoiceDataset):
    """The dataset of Qqp.

    The Quora Question Pairs2 dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent.

    Example:
        question1: How is the life of a math student? Could you describe your own experiences?
        question2: Which level of prepration is enough for the exam jlpt5?
        label: 0
    """

    instruction = "Determine whether following pair of questions are semantically equivalent.\nQuestion1: {{question1}}\nQuestion2: {{question2}}{{'\n'+options if options}}\nAnswer: "
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("nyu-mll/glue", "qqp")

    def format_instance(self, instance):
        instance["options"] = ["no", "yes"]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
