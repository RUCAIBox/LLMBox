from .multiple_choice_dataset import MultipleChoiceDataset
from datasets import load_dataset, load_from_disk


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

    def __init__(self, args):
        self.name = "copa"
        dataset = load_dataset("super_glue", "copa")
        # dataset = load_from_disk("copa")
        self.example_data = list(dataset[args.example_set])
        self.evaluation_data = list(dataset[args.evaluation_set])

        super().__init__(args)

    def format_instance(self, instance):
        source_text = instance["premise"][:-1]
        if instance["question"] == "cause":
            source_text += " because"
        elif instance["question"] == "effect":
            source_text += " therefore"

        label2text = {
            -1: "",
            0: " " + instance["choice1"][0].lower() + instance["choice1"][1:],
            1: " " + instance["choice2"][0].lower() + instance["choice2"][1:],
        }

        options = []
        for option in [0, 1]:
            target_text = label2text[option]
            options.append((source_text, target_text))
        return dict(ground_truth=(source_text, label2text[instance['label']]), options=options)

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
