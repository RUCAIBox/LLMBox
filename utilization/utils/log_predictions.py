import json
from logging import getLogger
from multiprocessing import Process, Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

logger = getLogger(__name__)


def repeat_iter(obj, n: int):
    """`repeat_iter([1, 2, 3], 2)` -> `[1, 2, 3, 1, 2, 3]`"""
    for _ in range(n):
        yield from obj


def to_dict(merge: Optional[List[str]] = None, merge_by_option: Optional[List[str]] = None):
    merge = merge or []
    merge_by_option = merge_by_option or []

    def wrapper(df: pd.DataFrame):
        df_dict = df.to_dict(orient="list")
        for col in merge:
            df_dict[col] = df_dict[col][0]
        if "option_num" in df_dict:
            option_num = df_dict.pop("option_num")[0]
            for col in merge_by_option:
                df_dict[col] = df_dict[col][:option_num]
        return df_dict

    return wrapper


class PredictionWriter:

    def __init__(self, evaluation_path: str):
        self.evaluation_path = evaluation_path
        self.queue = Queue()
        self.process = Process(target=PredictionWriter._listen_and_write, args=(self.queue, self.evaluation_path))
        self.process.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.queue.put("STOP")
        self.process.join()

    @staticmethod
    def _listen_and_write(queue: Queue, file: str):
        while True:
            data = queue.get()
            if data == "STOP":
                break
            try:
                with open(file, "a") as f:
                    json.dump(data, f, ensure_ascii=False)
                    f.write("\n")
            except Exception as e:
                logger.warning(f"Failed to log_predictions: {e}\n{data}")
                break

    def _write(self, data):
        self.queue.put(data)

    def log_batch_predictions(
        self,
        raw_predictions: List[str],
        lines_iter: Iterator[Tuple[int, str, Any]],
    ) -> int:
        if not self.process.is_alive():
            return 0

        for raw_prediction, (idx, source, reference) in zip(raw_predictions, lines_iter):
            lines = {
                "index": idx,
                "source": source,
                "raw_prediction": raw_prediction,
                "reference": reference,
            }
            self._write(lines)
        return len(raw_predictions)


def log_final_predictions(
    raw_predictions: List[str],
    processed_predictions: List[Union[str, float]],
    score_lists: Dict[str, List[float]],
    multiple_source: bool,
    model_evaluation_method: str,
    use_normalization: bool,
    option_nums: List[int],
    evaluation_data: List[Dict[str, Any]],
    evaluation_instances: List[tuple],
    sample_num: int,
    references: List[Any],
) -> Optional[pd.Series]:

    transposed_score_lists = [dict(zip(score_lists.keys(), values)) for values in zip(*score_lists.values())]
    if model_evaluation_method == "generation":
        # only generation tasks support self-consistency
        lines = {
            "index": repeat_iter(range(len(evaluation_data)), sample_num),
            "source": evaluation_instances,
            "raw_prediction": raw_predictions,
            "processed_prediction": processed_predictions,
            "reference": repeat_iter(references, sample_num),
            "metric": repeat_iter(transposed_score_lists, sample_num),
        }
        try:
            return pd.DataFrame(lines).groupby("index").apply(to_dict(merge=["index", "source", "metric", "reference"]))
        except Exception as e:
            lines = {k: len(v) for k, v in lines.items()}
            logger.warning(f"Failed to log_predictions: {e}\n{lines}")
            return None

    elif model_evaluation_method == "get_ppl":  # ranking

        def repeat_by_option(*arr):

            def wrapper():
                for cols in zip(range(len(option_nums)), *arr):
                    for _ in range(option_nums[cols[0]]):
                        yield (*cols, option_nums[cols[0]])

            return zip(*wrapper())

        source_text, target_text = zip(*evaluation_instances)
        if use_normalization:
            source_text, target_text, raw_predictions = source_text[::2], target_text[::2], raw_predictions[::2]
        index, references, transposed_score_lists, option_nums = repeat_by_option(references, transposed_score_lists)
        lines = {
            "index": index,
            "source": source_text,
            "option": target_text,
            "option_num": option_nums,
            "perplexity": map(lambda r: r[0], raw_predictions),
            "reference": references,
            "metric": transposed_score_lists,
        }
        try:
            if multiple_source:
                merge = ["index", "option", "reference", "metric"]
                merge_by_option = ["source"]
            else:
                merge = ["index", "source", "reference", "metric"]
                merge_by_option = ["option"]
            return pd.DataFrame(lines).groupby("index").apply(to_dict(merge, merge_by_option))
        except Exception as e:
            lines = {k: len(v) for k, v in lines.items()}
            logger.warning(f"Failed to log_predictions: {e}\n{lines}")
            return None

    elif model_evaluation_method == "get_prob":

        lines = {
            "index": range(len(evaluation_data)),
            "source": map(lambda i: i[0], evaluation_instances),
            "probabilites": raw_predictions,
            "prediction": processed_predictions,
            "reference": references,
            "metric": transposed_score_lists,
        }
        try:
            return pd.Series(lines)
        except Exception as e:
            lines = {k: len(v) for k, v in lines.items()}
            logger.warning(f"Failed to log_predictions: {e}\n{lines}")
            return None

    else:
        logger.debug(f"Failed to log predictions: processed_predictions={processed_predictions}")
        return None
