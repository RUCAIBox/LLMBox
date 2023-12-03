from typing import Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..utils import NotImplementedField


class Model:
    r"""The base model object for all models.

    Args:
        args (ModelArguments): The global configurations.

    Attributes:
        name (str): The name of this model.
        type (str): The type of this model, which can be chosen from `base` and `instruction`.
        tokenizer (Union[transformers.PreTrainedTokenizer, tiktoken.Encoding]): The tokenizer of this model.
        max_tokens (int): The maximum token length of this model.
        generation_kwargs (dict): The configurations for open-ended generation.
        ppl_kwargs (dict, *optional*): The configurations for computing PPL score.
    """
    name = ""
    type = NotImplementedField

    def __init__(self, args):
        self.args = args
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None

    def set_ppl_args(self, **kwargs):
        r"""Set the configurations for PPL score calculation."""

        raise NotImplementedError(f"{self.name} model must implement the `set_ppl_args` function.")

    def get_ppl(self, batched_inputs):
        r"""Compute the PPL score of the target text given the source text for this batch.

        Args:
            batched_inputs (List[Tuple(str, str)]): A list of tuples of source and target texts.

        Returns:
            List(float): The list of PPL scores.
        """
        raise NotImplementedError(f"{self.name} model must implement the `get_ppl` function.")

    def set_generation_args(self, **kwargs):
        r"""Set the configurations for open-ended generation."""

        raise NotImplementedError(f"{self.name} model must implement the `set_generation_args` function.")

    def generation(self, batched_inputs, generation_args: Optional[dict] = None):
        r"""Generate the response of given question for this batch.

        Args:
            batch (List[str]): The batch of questions.
            generation_args (dict, optional): The configurations for generation. It will first be merged with a `GenerationConfig`, and then all unused kwargs are passed as model kwargs. See `HuggingfaceModel.generation` for more huggingface-model-specific kwargs.

        Returns:
            List(str): The list of generation results.
        """
        raise NotImplementedError(f"{self.name} model must implement the `generation` function.")
