from .base import Base
from .method_oriented import get_prompt


class BaseCoT(Base):
    """
    A base class for implementing Chain of Thought (CoT) reasoning in models.

    This class serves as a foundational component for models that employ the CoT approach to process and answer queries. It sets up the basic structure and methods that are common across different CoT implementations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.examples = get_prompt(['chain_of_thought', 'cot_trigger']) + '\n'


class ZSCoT(BaseCoT):
    """
    A class for implementing Zero-Shot Chain of Thought (ZSCoT) reasoning.

    This class is designed for situations where no prior examples (zero-shot) are provided to the model. It utilizes the base CoT approach and extends it to work in a zero-shot learning environment.

    Paper Link: https://arxiv.org/pdf/2205.11916.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CoT(BaseCoT):
    """
    Paper
    A class for implementing Chain of Thought (CoT) reasoning with few-shot examples.

    This class enhances the base CoT approach by incorporating few-shot learning, where a small number of example cases are used to guide the model's reasoning process.

    Paper Link: https://arxiv.org/pdf/2201.11903.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.examples = get_prompt(['chain_of_thought', self.dataset_name]) + '\n'
