from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Cola(MultipleChoiceDataset):
    """The dataset of Cola.

    The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.

    Example:
        sentence: The man turned on the faucet.
        label: 1
    """

    instruction = "Determine whether the following sentence is an acceptable grammatical English sentence.\n\n{{sentence}}{{'\n'+options+'\nAnswer:' if options}}"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("nyu-mll/glue", "cola")

    def format_instance(self, instance):
        instance["options"] = [
            "unacceptable",
            "acceptable",
        ]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
