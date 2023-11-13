import re


def context_processor(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = re.sub(" +", " ", text)
    return text
