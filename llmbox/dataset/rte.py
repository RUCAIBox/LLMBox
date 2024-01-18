from .multiple_choice_dataset import MultipleChoiceDataset


class Rte(MultipleChoiceDataset):
    """The dataset of Rte

    Recognizing Textual Entailment (Rte) datasets come from a series of annual competitions on textual entailment. 

    Example:
        premise: It was a complex language. Not written down but handed down. One might say it was peeled down.
        passage: the language was peeled down
        label: 0
    """
    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "rte")

    def format_instance(self, instance):
        source = instance["premise"] + '\n' + 'question: ' + instance["hypothesis"
                                                                      ] + ' True or False?' + '\n' + 'answer:'

        label2text = {
            0: ' True',
            1: ' False',
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
