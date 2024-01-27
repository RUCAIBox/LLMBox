from .multiple_choice_dataset import MultipleChoiceDataset


class Cb(MultipleChoiceDataset):
    """The dataset of Cb

    CommitmentBank (Cb, de Marneffe et al., 2019) is a corpus of short texts in which at least one sentence contains an embedded clause.

    Example:
        premise: It was a complex language. Not written down but handed down. One might say it was peeled down.
        passage: the language was peeled down
        label: 0
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "cb")

    def format_instance(self, instance):
        source = instance["premise"] + "\nquestion: " + instance["hypothesis"] + ". true, false, or neither?\nanswer:"

        label2text = {
            0: " true",
            1: " false",
            2: " neither",
        }

        options = [label2text[option] for option in [0, 1, 2]]
        return dict(
            source=source,
            target=label2text[instance["label"]],
            options=options,
        )

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
