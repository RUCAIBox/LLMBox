from functools import cached_property
from typing import Optional

from ..metric import WordAccuracy
from .generation_dataset import GenerationDataset
from .dataset_utils.ruler_utils import 


class Lambada(GenerationDataset):
    r"""The dataset of lambada.

    The LAMBADA evaluates the capabilities of computational models for text understanding by means of a word prediction task. LAMBADA is a collection of narrative passages sharing the characteristic that human subjects are able to guess their last word if they are exposed to the whole passage, but not if they only see the last sentence preceding the target word. To succeed on LAMBADA, computational models cannot simply rely on local context, but must be able to keep track of information in the broader discourse.

    Example:
        'category': 'Mystery',
        'text': 'bob could have been called in at this point , but he was n't miffed at his exclusion at all . he was relieved at not being brought into this initial discussion with central command . `` let 's go make some grub , '' said bob as he turned to danny . danny did n't keep his stoic expression , but with a look of irritation got up and left the room with bob',

    """

    instruction = "{source}"
    evaluation_set = "test"
    example_set = None
    load_args = ("EleutherAI/lambada_openai", "default")
    extra_model_args = dict(max_tokens=5, temperature=0)
    metrics = [WordAccuracy()]

    def load_raw_dataset(self, dataset_path: Optional[str], subset_name: Optional[str], evaluation_set: str, example_set: Optional[str]):
        self.evaluation_data = 