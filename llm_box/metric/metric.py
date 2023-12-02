class Metric:
    r""" The base class for metric calculation.
    """

    def __call__(self, predictions, references):
        r""" Compute specific metric scores between predictions and references.

        Args:
            predictions (List[Union[str, int, float]]): The predicted answers.
            references (List[Union[str, int, float, List[str]]]): The predicted answers.

        Returns:
            Dict[str, float]: The metric score.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} metric must implement the `__call__` function for score calculation."
        )
