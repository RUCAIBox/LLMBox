class Model:
    r"""The base model object for all models.

    Args:
        args (Namespace): The global configurations.
    
    Attributes:
        name (str): The name of this model.
        tokenizer (Union[transformers.PreTrainedTokenizer, tiktoken.Encoding]): The tokenizer of this model.
        max_tokens (int): The maximum token length of this model.
        generation_kwargs (dict): The configurations for open-ended generation.
        ppl_kwargs (dict, *optional*): The configurations for computing PPL score.
    """

    def __init__(self, args):
        self.args = args

    def get_ppl(self, batch):
        r"""Compute the PPL score of the option given the context for this batch.

        Args:
            batch (List[Tuple(str, str)]): The batch of context and option pairs.
        
        Returns:
            List(float): The list of PPL scores.
        """
        raise NotImplementedError(f"{self.name} model must implement the `get_ppl` function.")

    def generation(self, batch):
        r"""Generate the response of given question for this batch.

        Args:
            batch (List[str]): The batch of questions.
        
        Returns:
            List(str): The list of generation results.
        """
        raise NotImplementedError(f"{self.name} model must implement the `generation` function.")
