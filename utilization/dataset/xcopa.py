from .multiple_choice_dataset import MultipleChoiceDataset


class Xcopa(MultipleChoiceDataset):
    """The dataset of Xcopa.

    XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across languages. The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around the globe. The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages. All the details about the creation of XCOPA and the implementation of the baselines are available in the paper.

    Example:
        premise: The man turned on the faucet.
        choice1: The toilet filled with water.
        choice2: Water flowed from the spout.
        question: effect
        label: 1
    """

    instruction = "Complete the following the sentence.\n\n{{premise[:-1]}}{{' because' if question == 'cause' else ' therefore'}}{{'\n'+options+'\nAnswer:' if options}}"
    evaluation_set = "validation"
    example_set = "test"
    load_args = ("xcopa",)

    def format_instance(self, instance):
        instance["options"] = [
            instance["choice1"][0].lower() + instance["choice1"][1:],
            instance["choice2"][0].lower() + instance["choice2"][1:],
        ]
        return instance

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
