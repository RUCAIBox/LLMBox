from .multiple_choice_dataset import MultipleChoiceDataset


class Rte(MultipleChoiceDataset):
    """The dataset of Rte

    Recognizing Textual Entailment (Rte) datasets come from a series of annual competitions on textual entailment.

    Example:
        premise: It was a complex language. Not written down but handed down. One might say it was peeled down.
        passage: the language was peeled down
        label: 0
    """

    instruction = "{{premise}}\nquestion: {{hypothesis}} True or False?{{'\n' + options if options}}\nanswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "rte")

    def format_instance(self, instance):
        instance["options"] = ["True", "False"]
        return instance

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
