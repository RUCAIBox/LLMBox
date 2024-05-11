from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Wic(MultipleChoiceDataset):
    """The dataset of Wic

    Word-in-Context (Wic, Pilehvar and Camacho-Collados, 2019) is a word sense disambiguation task cast as binary classification of sentence pairs.

    Example:
        word: place
        sentence1: Do you want to come over to my place later?
        sentence2: A political system with no place for the less prominent groups.
        start1: 31
        start2: 27
        end1: 36
        end2: 32
        label: 0
    """

    instruction = "{{sentence1}}\n{{sentence2}}\nquestion: Is the word '{{word}}' used in the same way in the two sentences above?{{'\n' + options if options}}\nanswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "wic")

    def format_instance(self, instance):
        instance["options"] = [" no", " yes"]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
