from functools import cached_property

from ..metric import Rouge
from .generation_dataset import GenerationDataset


class CNN_DailyMail(GenerationDataset):
    """The dataset of cnn_dailymail.

    The CNN / DailyMail Dataset is an English-language dataset containing
    just over 300k unique news articles as written by journalists at CNN
    and the Daily Mail, supports both extractive and abstractive summarization.

    Examples:
        article: (CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour.
        highlights: The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says .
    """

    instruction = "{article}\n\nTL;DR:"
    evaluation_set = "test"
    example_set = "train"
    metrics = [Rouge()]
    load_args = ("cnn_dailymail", "3.0.0")
    extra_model_args = dict(temperature=0)

    def format_instance(self, instance):
        return dict(article=instance["article"], target=instance["highlights"])

    @cached_property
    def references(self):
        return [instance["highlights"][:] for instance in self.evaluation_data]
