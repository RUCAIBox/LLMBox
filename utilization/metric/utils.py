from typing import Dict, List, Literal, Optional

import numpy as np

# To calculate the overall metrics for some datasets (e.g. AGIEval)
# which includes both GenerationDataset and MultipleChoiceDataset
METRIC_ALIASES = {"Accuracy": "EM"}
REVERSED_ALIASES = {v: k for k, v in METRIC_ALIASES.items()}


def avg_metrics(
    subset_results: List[Dict[str, float]],
    weighted_factor: Optional[List[int]] = None,
    average_method: Literal["macro", "weighted"] = "macro"
) -> Optional[Dict[str, float]]:
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
    if len(subset_results) <= 1:
        return None
    metric_entries = next(iter(subset_results)).keys()

    for metric in metric_entries:
        used_metrics = [metric]

        def i(results):
            if metric in results:
                return results[metric]
            elif metric in METRIC_ALIASES and METRIC_ALIASES[metric] in results:
                used_metrics.append(METRIC_ALIASES[metric])
                return results[METRIC_ALIASES[metric]]
            elif metric in REVERSED_ALIASES and REVERSED_ALIASES[metric] in results:
                used_metrics.append(REVERSED_ALIASES[metric])
                return results[REVERSED_ALIASES[metric]]
            else:
                raise KeyError(f"Metric {metric} not found in the results.")

        str_m = "/".join(used_metrics)
        if average_method == "macro":
            results[str_m] = np.mean([i(r) for r in subset_results])
        elif average_method == "weighted":
            results[str_m] = np.sum([i(r) * c
                                     for r, c in zip(subset_results, weighted_factor)]) / np.sum(weighted_factor)

    return results
