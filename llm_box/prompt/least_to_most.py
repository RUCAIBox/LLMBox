from .method_oriented import get_prompt
from .base import Base


class LeastToMost(Base):
    """
    A class designed to implement the 'least-to-most' prompting strategy in AI models.

    This class utilizes a step-by-step approach, breaking down complex problems into simpler sub-problems. It leverages few-shot examples tailored to specific datasets to guide the model in sequentially addressing each part of a problem, ultimately leading to a comprehensive solution.

    Paper Link: https://arxiv.org/pdf/2205.10625.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.examples = get_prompt(['least_to_most', self.dataset_name]) + '\n'
