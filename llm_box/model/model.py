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
        self.tokenizer = None

    def get_ppl(self, batched_inputs):
        r"""Compute the PPL score of the option given the context for this batch.

        Args:
            batched_inputs (List[Tuple(str, str)]): The batch of context and option pairs.

        Returns:
            List(float): The list of PPL scores.
        """
        raise NotImplementedError(f"{self.name} model must implement the `get_ppl` function.")

    def generation(self, batched_inputs):
        r"""Generate the response of given question for this batch.

        Args:
            batched_inputs (List[str]): The batch of questions.

        Returns:
            List(str): The list of generation results.
        """
        raise NotImplementedError(f"{self.name} model must implement the `generation` function.")
