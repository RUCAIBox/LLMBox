from .multiple_choice_dataset import MultipleChoiceDataset


class Wic(MultipleChoiceDataset):
    """The dataset of Wic

    Word-in-Context (Wic, Pilehvar and Camacho-Collados, 2019) is a word sense disambiguation task cast as binary classification of sentence pairs. 

    Example:
        word: place
        sentence1: Do you want to come over to my place later?
        sentence2: A political system with no place for the less prominent groups.
        start1: 31
        start2: 27
        end1: 36
        end2: 32
        label: 0
    """
    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "wic")

    def format_instance(self, instance):
        source = instance["sentence1"] + '\n' + instance[
            "sentence2"
        ] + '\n' + f"question: Is the word '{instance['word']}' used in the same way in the two sentences above?\nanswer:"

        label2text = {
            0: ' no',
            1: ' yes',
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
