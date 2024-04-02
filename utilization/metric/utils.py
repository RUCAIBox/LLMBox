from typing import Dict, List, Literal, Optional

import numpy as np


def avg_metrics(
    subset_results: List[Dict[str, float]],
    weighted_factor: Optional[List[int]] = None,
    average_method: Literal["macro", "weighted"] = "macro"
) -> Dict[str, float]:
    """Calculate the average of the metrics in the subset_results.

    Args:
        subset_results (`List[Tuple[Dict[str, float], int]]`): A list of tuples containing the results and the count of each subset.
        average_method (`Literal["macro", "weighted"]`): The method to calculate the average.

    Returns:
        Dict[str, float]: A dictionary containing the average of the metrics in the subset_results.
    """
    if average_method == "weighted":
        assert weighted_factor is not None, "weighted_factor must be provided when average_method is 'weighted'."
    results = {}
    metric_entries = next(iter(subset_results)).keys()
    alias = {"Accuracy": "EM"}
    reversed_alias = {v: k for k, v in alias.items()}

    for metric in metric_entries:
        if metric in alias.keys():
            fallback = metric
            metric = alias[metric]
        elif metric in reversed_alias.keys():
            fallback = reversed_alias[metric]
        else:
            fallback = None

        def i(results):
            return results[metric] if metric in results else results[fallback]

        str_m = metric if fallback is None else metric + "/" + fallback
        if average_method == "macro":
            results[str_m] = np.mean([i(r) for r in subset_results])
        elif average_method == "weighted":
            results[str_m] = np.sum([i(r) * c
                                     for r, c in zip(subset_results, weighted_factor)]) / np.sum(weighted_factor)

    print(results, metric_entries)
    return results
