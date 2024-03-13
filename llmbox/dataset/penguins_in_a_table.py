from .multiple_choice_dataset import MultipleChoiceDataset
from ..metric import Accuracy
import numpy as np

class Penguins_in_a_table(MultipleChoiceDataset):
    """The dataset of Copa.

    The Choice Of Plausible Alternatives (COPA, Roemmele et al., 2011) dataset is a causal reasoning task in which a system is given a premise sentence and two possible alternatives.

    Example:
        premise: The man turned on the faucet.
        choice1: The toilet filled with water.
        choice2: Water flowed from the spout.
        question: effect
        label: 1
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("tasksource/bigbench", "penguins_in_a_table")
    metrics = [Accuracy()]

    def format_instance(self, instance):
        source = instance["inputs"]
        options = [" " + option for option in instance["multiple_choice_targets"]]
        return dict(
            source=source,
            source_postfix="",
            target_idx=int(np.argmax(instance["multiple_choice_scores"])),
            options=options,
        )

    @property
    def references(self):
        return [int(np.argmax(instance["multiple_choice_scores"])) for instance in self.evaluation_data]
