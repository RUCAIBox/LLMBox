from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Copa(MultipleChoiceDataset):
    """The dataset of Copa.

    The Choice Of Plausible Alternatives (COPA, Roemmele et al., 2011) dataset is a causal reasoning task in which a system is given a premise sentence and two possible alternatives.

    Example:
        premise: The man turned on the faucet.
        choice1: The toilet filled with water.
        choice2: Water flowed from the spout.
        question: effect
        label: 1
    """

    instruction = "Complete the following the sentence.\n\n{{premise[:-1]}}{{' because' if question == 'cause' else ' therefore'}}{{'\n'+options+'\nAnswer:' if options}}"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "copa")

    def format_instance(self, instance):
        instance["options"] = [
            instance["choice1"][0].lower() + instance["choice1"][1:],
            instance["choice2"][0].lower() + instance["choice2"][1:],
        ]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
