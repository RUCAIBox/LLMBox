from .base import Base
from .least_to_most import LeastToMost
from .chain_of_thought import ZSCoT, CoT
from .pal import PAL

SUPPORTED_METHODS = ['pal', 'CoT', 'ZSCoT', 'least_to_most', 'baseline']

METHOD_MAP = {
    'pal': PAL,
    'CoT': CoT,
    'ZSCoT': ZSCoT,
    'least_to_most': LeastToMost,
    'baseline': Base,
}

METHOD_SUPPORT_DATASET = {
    'pal': ['gsm8k'],
    'CoT': ['gsm8k', 'csqa', 'bigbench_date', 'bigbench_object_tracking'],
    'ZSCoT': ['gsm8k', 'csqa', 'bigbench_date', 'bigbench_object_tracking'],
    'least_to_most': ['gsm8k', 'last_letter_concat'],
}


class PEMethod(object):
    """
    A class that provides an interface for various methods in prompt engineering.
    """

    def __init__(self, **kwargs):
        self.method = kwargs.get('method')
        self.infer_method = self.create_method(**kwargs)

    def create_method(self, **kwargs):
        """Creates and returns the appropriate method based on the method name."""

        # Get the method class based on the method name and instantiate it
        method_class = METHOD_MAP.get(self.method)
        if method_class:
            return method_class(**kwargs)
        else:
            raise ValueError("The method is not supported!")

    @staticmethod
    def method_list():
        """Returns a list of supported methods."""
        return METHOD_MAP.keys()

