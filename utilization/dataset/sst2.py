from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Sst2(MultipleChoiceDataset):
    """The dataset of Sst2.

    The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.

    Example:
        sentence: hide new secretions from the parental units
        label: 1
    """

    instruction = "Determine the sentiment of the following sentence.\n{{sentence.strip()}}{{ '\n' + options if options }}\nAnswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("nyu-mll/glue", "sst2")

    def format_instance(self, instance):
        instance["options"] = ["negative", "positive"]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
