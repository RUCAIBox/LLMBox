import re
from logging import getLogger

import torch
from prefetch_generator import BackgroundGenerator

logger = getLogger(__name__)


def context_processor(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = re.sub(" +", " ", text)
    return text


@property
def NotImplementedField(self):
    raise NotImplementedError(f"{self.__class__.__name__} has not implemented field.")


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

