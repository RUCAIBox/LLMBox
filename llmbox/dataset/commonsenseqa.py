from .multiple_choice_dataset import MultipleChoiceDataset


class Commonsenseqa(MultipleChoiceDataset):
    """The dataset of CommonsenseQA

    CommonsenseQA(Alon Talmor,2019) is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers .

    Example:
        question: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
        choices: { "label": [ "A", "B", "C", "D", "E" ], "text": [ "ignore", "enforce", "authoritarian", "yell at", "avoid" ] }
        answerKey: A
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("commonsense_qa",)

    def format_instance(self, instance):
        source = "Question: " + instance["question"] + "\n"
        source += "Answer:"
        label2text = {
            "A": " " + instance["choices"]["text"][0],
            "B": " " + instance["choices"]["text"][1],
            "C": " " + instance["choices"]["text"][2],
            "D": " " + instance["choices"]["text"][3],
            "E": " " + instance["choices"]["text"][4],
        }

        options = [label2text[option] for option in ["A", "B", "C", "D", "E"]]
        return dict(
            source=source,
            target=label2text[instance["answerKey"]],
            options=options,
        )

    @property
    def references(self):
        return [
            ord(instance["answerKey"]) - ord("A") for instance in self.evaluation_data
        ]
