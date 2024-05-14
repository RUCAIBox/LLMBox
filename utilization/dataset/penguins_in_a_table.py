from functools import cached_property

import numpy as np

from ..metric import Accuracy
from .multiple_choice_dataset import MultipleChoiceDataset


class Penguins_in_a_table(MultipleChoiceDataset):
    """The dataset of Bigbench Penguins_in_a_table.

    BIG-Bench but it doesn't require the hellish dependencies (tensorflow, pypi-bigbench, protobuf) of the official version.

    Example:
        inputs: Here is a table where the first line is a header and each subsequent line is a penguin: name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15 For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm. We now add a penguin to the table: James, 12, 90, 12 And here is a similar table, but listing giraffes: name, age, height (cm), weight (kg) Jody, 5, 430, 620 Gladys, 10, 420, 590 Marian, 2, 310, 410 Donna, 9, 440, 650 Which is the oldest penguin? Answer:
        multiple_choice_targets: [ "Louis", "Bernard", "Vincent", "Gwen", "James" ]
        multiple_choice_scores: [ 0, 0, 0, 0, 1 ]
    """

    instruction = "{{inputs}}{{'\n' + options if options}}"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("tasksource/bigbench", "penguins_in_a_table")
    metrics = [Accuracy()]

    def format_instance(self, instance):
        instance["arget_idx"] = int(np.argmax(instance["multiple_choice_scores"]))
        instance["options"] = instance["multiple_choice_targets"]
        return instance

    @cached_property
    def references(self):
        return [int(np.argmax(instance["multiple_choice_scores"])) for instance in self.evaluation_data]
