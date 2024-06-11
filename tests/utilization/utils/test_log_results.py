import json
import sys
from copy import deepcopy

import pandas as pd

from ..fixtures import *

sys.path.append('.')
from utilization.model.model_utils.conversation import Conversation, ConversationFormatter
from utilization.utils.log_results import log_final_results


def get_conv(split):
    if split:
        msg = [{
            "role": "user",
            "content": "This is the example input of the model."
        }, {
            "role": "assistant",
            "content": "This is the sample output of the model."
        }, {
            "role": "user",
            "content": "This is the input of the model."
        }]
    else:
        msg = [{"role": "user", "content": "This is the input of the model."}]
    return Conversation(
        msg,
        formatter=ConversationFormatter.from_chat_template("base"),
        model_evaluation_method="generation",
        split=split,
    )


data = {
    ("generation", "no_split", "legacy"): [("This is the input of the model.",)],
    ("generation", "split", "legacy"): [("This is", " a splitted sentence.")],
    ("generation", "split", "conv"): [get_conv(True)],
    ("generation", "no_split", "conv"): [get_conv(False)],
    ("get_ppl", "no_split", "legacy"): [("Source parts of get_ppl", " target parts 1 of get_ppl"),
                                        ("Source parts of get_ppl", " target parts 2 of get_ppl")],
    ("get_ppl", "split", "legacy"):
    [("Source parts of get_ppl", " can be splitted, but not", " target parts 1 of get_ppl"),
     ("Source parts of get_ppl", " can be splitted, but not", " target parts 2 of get_ppl")],
    ("get_prob", "no_split", "legacy"): [("This is the get_prob input of the model", 2)],
    ("get_prob", "split", "legacy"): [("The get_prob input of the model", " can also be splitted.", 2)],
}
methods = [
    "generation:no_norm:legacy:sample1:local",
    "generation:no_norm:conv:sample1:api",
    "generation:no_norm:conv:sample1:local",
    "generation:no_norm:legacy:sample2:local",
    "generation:no_norm:conv:sample2:api",
    "generation:no_norm:conv:sample2:local",
    "get_prob:no_norm:legacy:sample1:local",
    "get_ppl:no_norm:legacy:sample1:local",
    "get_ppl:acc_norm:legacy:sample1:local",
]


@pytest.mark.parametrize("split", ["split", "no_split"])
@pytest.mark.parametrize("multiple_source", [True, False])
@pytest.mark.parametrize("methods", methods)
def test_log_final_results(split, multiple_source, methods):

    eval_method, use_normalization, use_conversation, sample_num, local = methods.split(":")
    use_normalization = use_normalization == "acc_norm"
    sample_num = int(sample_num[-1])

    def set_subset(l: dict):
        l["subset"] = "subset_name"

    eval_data = data[eval_method, split, use_conversation]
    if eval_method == "get_ppl":
        raw = [(0.5, 10), (1.0, 10)]  # probabilities, length
        processed = [1]  # index 0
        op_num = 2
    elif eval_method == "get_prob":
        raw = [[0.1, 0.9]]  # probabilities
        processed = [1]  # index 0
        op_num = 2
    elif eval_method == "generation":
        raw = ["This is the model's raw prediction."]
        processed = ["prediction"]
        op_num = 1

    if use_normalization:
        no_num = 2
    else:
        no_num = 1

    series = log_final_results(
        raw_predictions=raw * sample_num * no_num,
        processed_predictions=processed * sample_num,
        evaluation_instances=deepcopy(eval_data) * sample_num * no_num,
        score_lists={"Metric": [True]},  # score_lists have already been aggregated along self-concsistency
        multiple_source=multiple_source,
        model_evaluation_method=eval_method,
        use_normalization=use_normalization,
        option_nums=[op_num] * sample_num,
        len_evaluation_data=1,
        sample_num=sample_num,
        references=["reference"],
        local_model=local == "local",
    )
    series.apply(set_subset)
    print(series)
    json_str = pd.concat([series]).to_json(orient="records", indent=4, force_ascii=False)

    unmarsheled = json.loads(json_str)
    print(json_str)
    print(unmarsheled)
    assert len(unmarsheled) == 1
    assert unmarsheled[0]["index"] == 0
    assert unmarsheled[0]["subset"] == "subset_name"
    if eval_method == "get_ppl" and not multiple_source:
        source = "".join(eval_data[0][:-1])
        assert unmarsheled[0]["source"] == source
        assert unmarsheled[0]["option"] == [" target parts 1 of get_ppl", " target parts 2 of get_ppl"]
        assert unmarsheled[0]["perplexity"] == [0.5, 1.0]
    elif eval_method == "get_ppl" and multiple_source:
        source = "".join(eval_data[0][:-1])
        assert unmarsheled[0]["source"] == [source, source]
        assert unmarsheled[0]["option"] == " target parts 1 of get_ppl"
        assert unmarsheled[0]["perplexity"] == [0.5, 1.0]
    elif eval_method == "get_prob":
        source = "".join(eval_data[0][:-1])
        assert unmarsheled[0]["source"] == source
        assert unmarsheled[0]["probabilites"] == [0.1, 0.9]
    elif eval_method == "generation":
        if use_conversation == "conv" and local == "local":
            source = eval_data[0].apply_prompt_template()
        elif use_conversation == "conv" and local == "api":
            source = eval_data[0].messages
        else:
            source = "".join(eval_data[0])
        assert unmarsheled[0]["source"] == source
        assert unmarsheled[0]["raw_prediction"] == ["This is the model's raw prediction."] * sample_num
    assert unmarsheled[0]["reference"] == "reference"
    assert unmarsheled[0]["metric"]["Metric"] == True
