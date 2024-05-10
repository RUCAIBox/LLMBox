from .multiple_choice_dataset import MultipleChoiceDataset


class Commonsenseqa(MultipleChoiceDataset):
    """The dataset of CommonsenseQA

    CommonsenseQA(Alon Talmor,2019) is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers .

    Example:
        question: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
        choices: { "label": [ "A", "B", "C", "D", "E" ], "text": [ "ignore", "enforce", "authoritarian", "yell at", "avoid" ] }
        answerKey: A
    """

    instruction = "Question: {{question}}{{'\n' + options if options}}\nAnswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("commonsense_qa",)

    def format_instance(self, instance):
        instance["options"] = instance["choices"]["text"]
        instance["target_idx"] = ord(instance["answerKey"]) - 65
        return instance

    @property
    def references(self):
        return [ord(instance["answerKey"]) - ord("A") for instance in self.evaluation_data]
