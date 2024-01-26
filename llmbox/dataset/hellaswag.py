from typing import Dict, Any

from .multiple_choice_dataset import MultipleChoiceDataset


class Hellaswag(MultipleChoiceDataset):
    """The dataset of hellaswag.

    HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers et al., 2019)
    Hellaswag is a new dataset for commonsense NLI. The paper was published at ACL2019.

    Example:
        activity_label: Removing ice from car,
        ctx: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then,
        endings: "[\", the man adds wax to the windshield and cuts it.\", \", a person board a ski lift, while two men supporting the head of the per...",
        label: '3',
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("hellaswag", "neutral")

    def format_instance(self, instance):
        source = instance["activity_label"] + ": " + instance["ctx"]

        label2text: dict[int, str | Any] = {
            0: " " + instance["endings"][0][0].lower() + instance["endings"][0][1:],
            1: " " + instance["endings"][1][0].lower() + instance["endings"][1][1:],
            2: " " + instance["endings"][2][0].lower() + instance["endings"][2][1:],
            3: " " + instance["endings"][3][0].lower() + instance["endings"][3][1:],
        }

        options = [label2text[option] for option in [0, 1, 2, 3]]
        return dict(
            source=source,
            target=label2text[int(instance["label"])],
            options=options,
        )

    @property
    def references(self):
        return [int(instance["label"]) for instance in self.evaluation_data]
