from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Cola(MultipleChoiceDataset):
    """The dataset of Cola.

    The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.

    Example:
        sentence: The man turned on the faucet.
        choice1: 0 (unacceptable)
        choice2: 1 (acceptable)
        label: 1
    """

    instruction = "Judge whether the following sentence is an acceptable grammatical English sentence or not.\nsentence: {{source}}{{'\n'+options+'\nAnswer:' if options}}"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("nyu-mll/glue", "cola")

    def format_instance(self, instance):
        options = ["acceptable", "unacceptable"]
        return dict(
            source=instance["sentence"].strip(),
            target_id=instance["label"],
            options=options,
            idx=instance["idx"],
        )

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
