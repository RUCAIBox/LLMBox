import json
import typing
from dataclasses import asdict
from logging import getLogger
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

logger = getLogger(__name__)

if typing.TYPE_CHECKING:
    from .arguments import DatasetArguments, EvaluationArguments, ModelArguments


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


def dump_conversations(convs: List[Any], local: bool):
    from ..model.model_utils.conversation import Conversation

    if isinstance(convs, (str, Conversation)):
        convs = [convs]

    if isinstance(convs[0], Conversation):
        if not local:
            convs = [p.messages for p in convs]
        else:
            convs = [p.apply_prompt_template() for p in convs]

    return convs


class PredictionWriter:

    def __init__(self, evaluation_path: Optional[str]):
        self.evaluation_path = evaluation_path
        self._alive = isinstance(evaluation_path, str)

    def write_metainfo(
        self,
        model_args: "ModelArguments",
        dataset_args: "DatasetArguments",
        evaluation_args: "EvaluationArguments",
        continue_from: Optional[str] = None,
        load_continue_from: bool = True,
    ):
        self.model_args = model_args
        self.dataset_args = dataset_args
        self.evaluation_args = evaluation_args
        if self.alive():
            with open(self.evaluation_path, "w") as f:
                metainfo = {
                    "evaluation_results": "batch",
                    "model_args": asdict(model_args),
                    "dataset_args": asdict(dataset_args),
                    "evaluation_args": asdict(evaluation_args),
                }
                f.write(json.dumps(metainfo, ensure_ascii=False) + "\n")

        self.continue_from_instance = None
        if not load_continue_from:
            self.continue_from_path = None
            return

        # load num instances
        self.continue_from_path = continue_from or self.evaluation_args.continue_from
        if self.continue_from_path:
            self.continue_from_instance = self.check_continue()

        # set num instances in dataset_args
        if self.continue_from_instance is not None and continue_from is None:
            self.dataset_args.continue_from = self.continue_from_instance

    def alive(self):
        return self._alive

    def _write(self, data):
        try:
            with open(self.evaluation_path, "a") as f:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            logger.warning(f"Failed to log batch predictions: {e}\n{data}")
            self._alive = False

    def log_batch_results(
        self,
        raw_predictions: List[str],
        local_model: bool,
        lines_iter: Iterator[Tuple[int, str, Any]],
    ) -> int:
        """Log the batch predictions to the evaluation jsonlines file."""
        if not self.alive():
            return len(raw_predictions)

        for raw_prediction, (idx, source, reference) in zip(raw_predictions, lines_iter):
            source = dump_conversations(source, local_model)
            lines = {
                "index": idx,
                "source": source,
                "raw_prediction": raw_prediction,
                "reference": reference,
            }
            self._write(lines)
        return len(raw_predictions)

    def _parse_log_results(self, logline: Dict[str, typing.Any], write_back: bool = True) -> typing.Any:
        if write_back:
            self._write(logline)
        return logline["raw_prediction"]

    def check_continue(self) -> Optional[int]:

        def _check_args(metainfo: Dict[str, typing.Any]):
            pass

        with open(self.continue_from_path, "r") as f:
            iterator = iter(f)
            metainfo = next(iterator)
            if metainfo == "[":
                return None
            metainfo = json.loads(metainfo)
            num_lines = 0
            if "evaluation_results" in metainfo:
                _check_args(metainfo)
            else:
                num_lines += 1
            num_lines += sum(1 for _ in iterator)
            logger.info(f"Continue from {self.continue_from_path} ({num_lines} lines)")
            return num_lines

    def load_continue(self) -> Iterator[typing.Any]:

        assert self.continue_from_instance is not None
        with open(self.continue_from_path, "r") as f:
            iterator = iter(f)
            metainfo = json.loads(next(iterator))
            if "evaluation_results" not in metainfo:
                yield self._parse_log_results(metainfo)
            for line in f:
                yield self._parse_log_results(json.loads(line))


def _warn(e, lines):
    get_len = lambda v: len(v) if hasattr(v, "__len__") else None
    lines = {k: get_len(v) for k, v in lines.items()}
    error = e.__class__.__name__ + ": " + str(e)
    logger.warning(f"Failed to generate final predictions: {error}\n{lines}", stacklevel=2)


def log_final_results(
    raw_predictions: List[str],
    processed_predictions: List[Union[str, float]],
    evaluation_instances: List[tuple],
    score_lists: Dict[str, List[float]],
    multiple_source: bool,
    model_evaluation_method: str,
    use_normalization: bool,
    option_nums: List[int],
    len_evaluation_data: int,
    sample_num: int,
    references: List[Any],
    local_model: bool,
) -> Optional[pd.Series]:
    """Aggregate the final results and prepare for dumping to a json file."""

    evaluation_instances = dump_conversations(evaluation_instances, local_model)

    transposed_score_lists = [dict(zip(score_lists.keys(), values)) for values in zip(*score_lists.values())]
    if model_evaluation_method == "generation":
        if isinstance(evaluation_instances[0], tuple):
            evaluation_instances = ["".join(p) for p in evaluation_instances]

        # only generation tasks support self-consistency
        lines = {
            "index": list(repeat_iter(range(len_evaluation_data), sample_num)),
            "source": evaluation_instances,
            "raw_prediction": raw_predictions,
            "processed_prediction": processed_predictions,
            "reference": list(repeat_iter(references, sample_num)),
            "metric": list(repeat_iter(transposed_score_lists, sample_num)),
        }
        try:
            return pd.DataFrame(lines).groupby("index").apply(to_dict(merge=["index", "source", "metric", "reference"]))
        except Exception as e:
            _warn(e, lines)

    elif model_evaluation_method == "get_ppl":  # ranking

        def repeat_by_option(*arr):

            def wrapper():
                for cols in zip(range(len(option_nums)), *arr):
                    for _ in range(option_nums[cols[0]]):
                        yield (*cols, option_nums[cols[0]])

            return zip(*wrapper())

        *source_texts, target_text = zip(*evaluation_instances)
        source_text = ["".join(seg[sent_idx] for seg in source_texts) for sent_idx in range(len(source_texts[0]))]
        if use_normalization:
            src_len = len(source_text) // 2
            source_text, target_text, raw_predictions = source_text[:src_len], target_text[:src_len
                                                                                           ], raw_predictions[:src_len]
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
            _warn(e, lines)

    elif model_evaluation_method == "get_prob":

        lines = {
            "index": list(range(len_evaluation_data)),
            "source": list(map(lambda i: "".join(i[:-1]), evaluation_instances)),
            "probabilites": raw_predictions,
            "prediction": processed_predictions,
            "reference": references,
            "metric": transposed_score_lists,
        }
        try:
            return pd.DataFrame(lines).groupby("index").apply(to_dict(merge=lines.keys()))
        except Exception as e:
            _warn(e, lines)

    else:
        logger.debug(f"Failed to log predictions: processed_predictions={processed_predictions}")
    return None
