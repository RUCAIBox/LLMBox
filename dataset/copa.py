from datasets import load_dataset, load_from_disk

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

    def __init__(self, args, model):
        self.name = "copa"
        # dataset = load_dataset("super_glue", "copa")
        dataset = load_from_disk("../dataset/copa")
        self.example_data = list(dataset[args.example_set])
        self.evaluation_data = list(dataset[args.evaluation_set])
        self.instruction = "Complete the following the sentence."

        super().__init__(args, model)

    def format_instance(self, instance):
        source = instance["premise"][:-1]
        if instance["question"] == "cause":
            source += " because"
        elif instance["question"] == "effect":
            source += " therefore"

        label2text = {
            0: " " + instance["choice1"][0].lower() + instance["choice1"][1:],
            1: " " + instance["choice2"][0].lower() + instance["choice2"][1:],
        }

        options = [label2text[option] for option in [0, 1]]
        return dict(
            source=source,
            target=label2text[instance['label']],
            options=options,
        )

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
