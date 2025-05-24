from typing import Optional
from ..metric import PPL
from .validation_perplexity_dataset import ValidationPerplexityDataset


class WikiTextPPL(ValidationPerplexityDataset):
    r"""The dataset of wikitext ppl.

    Example:
        'text': 'It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .',

    """

    evaluation_set = "test"
    load_args = ("Salesforce/wikitext", "wikitext-2-raw-v1")
    metrics = [PPL()]

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, example_set)
        self.evaluation_data = [{"text": o["text"].strip()} for o in self.evaluation_data if len(o["text"].strip()) > 0]

    def format_instance(self, instance):
        return {
            "text": instance["text"],
            "options": [instance["text"]],
        }

