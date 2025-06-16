from functools import cached_property

from ..metric import WordAccuracy
from .generation_dataset import GenerationDataset

class Swde(GenerationDataset):
    r"""The dataset of SWDE.

    SWDE (Information Extraction). The task in the SWDE benchmark is to extract semi-structured relations from raw HTML websites. For example, given an IMBD page for a movie (e.g. Harry Potter and the Sorcerer’s Stone) and a relation key (e.g. release date), the model must extract the correct relation value (e.g. 2001). The SWDE benchmark was originally curated by Lockard et al. for the task of open information extraction from the semi-structured web.

    Example:
        'key': 'year',
        'value': '2005',
        'text': 'Tim Burton's Corpse Bride Movie Facts and Details click here amc home | movie guide Genres
        Lists
        Ratings amctv.com>movie guide>Tim Burton's Corpse Bride>details Tim Burton's Corpse Bride details
        Overall Rating Total Ratings: 1 Overview
        Details
        Cast & Credits
        Awards
        Review Movie Details: Director: Tim Burton, Mike Johnson
        Produced By: Will Vinton Studios, Warner Brothers Feature Animation, Tim Burton Animation Co
        Year: 2005
        Run Time: 76 minutes
        Country: UK, USA
        Language: English MPAA Rating: PG (for some scary images and action and brief mild language)
        Category: Animated
        Genre/Type: Fantasy
        Filmed In: Stop-Motion, Color
        Release: 2005 09 23 (USA), 2005 09 16 (USA - Limited) Alternate Titles: Corpse Bride
        Key Cast: Johnny Depp, Helena Bonham Carter, Emily Watson, Tracey Ullman, Albert Finney, Danny Elfman, Christopher (...)

        Summary of information above...
        genre/type: Fantasy
        director: Tim Burton, Mike Johnson
        run time: 76 minutes
        year:',

    """
    instruction = "{source}"
    evaluation_set = "validation"
    example_set = None
    load_args = ("hazyresearch/based-swde", "default")
    extra_model_args = dict(max_tokens=48, temperature=0, stop=["\n"])
    metrics = [WordAccuracy()]


    def format_instance(self, instance):
        instance["source"] = instance["text"]
        instance["target"] = instance["value"]
        return instance

    def post_processing(self, predictions):
        return [prediction.strip() for prediction in predictions]
    
    @cached_property
    def references(self):
        return [instance["target"] for instance in self.evaluation_data]