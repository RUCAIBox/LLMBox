from .multiple_choice_dataset import MultipleChoiceDataset


class Cb(MultipleChoiceDataset):
    """The dataset of Cb

    CommitmentBank (Cb, de Marneffe et al., 2019) is a corpus of short texts in which at least one sentence contains an embedded clause.

    Example:
        premise: It was a complex language. Not written down but handed down. One might say it was peeled down.
        passage: the language was peeled down
        label: 0
    """

    instruction = "{{premise}}\nquestion: {{hypothesis}}. true, false, or neither?{{'\n' + options if options}}\nanswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "cb")

    def format_instance(self, instance):
        instance["options"] = ["true", "false", "neither"]
        return instance

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
