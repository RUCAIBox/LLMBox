from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class Xwinograd(MultipleChoiceDataset):
    """The dataset of XWinograd.

    Multilingual winograd schema challenge as used in Crosslingual Generalization through Multitask Finetuning.
    
    Example:
        "sentence": "The city councilmen refused the demonstrators a permit because _ feared violence.",
        "option1": "the demonstrators",
        "option2": "The city councilmen",
        "answer": 2

    Note: 
        If you encounter network connectivity issues, we recommend copying this dataset and replacing the URL below in 'Muennighoff/xwinograd/xwinograd.py' with an accessible mirror site URL.
        Currently, we are unable to provide alternative solutions for network issues.
        Original: _URL = "https://huggingface.co/datasets/Muennighoff/xwinograd/raw/main/test/{lang}.jsonl"
        Mirror site: _URL = "https://hf-mirror.com/datasets/Muennighoff/xwinograd/raw/main/test/{lang}.jsonl"
    """

    instruction = "Given the sentence '{{sentence.strip()}}' in {{lang}}, fill in the blank with the appropriate option: who does '_' refer to?{{'\n'+options if options}}\nAnswer:"
    evaluation_set = "test"
    load_args = ("Muennighoff/xwinograd",)

    def init_arguments(self):
        from langcodes import Language
        self.language = Language(self.subset_name).language_name("en")

    def format_instance(self, instance):
        instance["lang"] = self.language
        instance["label"] = int(instance["answer"]) - 1
        instance["options"] = [instance["option1"], instance["option2"]]
        return instance

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
