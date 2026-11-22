import pytest

from utilization.dataset.gaokao import Gaokao
from utilization.metric.gaokao_bench_metric import GaokaoBenchMetric


def reference(answers, score=4, task="multi_answers_mcq"):
    return {"answer": answers, "task": task, "score": score}


@pytest.mark.parametrize(
    "selected,accepted", [
        ("AA", "AB"),
        ("AAB", "AB"),
        ("ABA", "AB"),
        ("ABBA", "AB"),
        ("CABAC", "ABC"),
        ("A", "AB"),
        ("AB", "ABC"),
        ("AAA", "ABC"),
        ("ABB", "ABC"),
        ("AB", "AB"),
        ("BA", "AB"),
        ("AAAA", "A"),
        ("C", "AB"),
        ("AAC", "AB"),
        ("CC", "AB"),
        ("", "AB"),
        ("A?", "AB"),
        ("A B", "AB"),
        ("ab", "AB"),
    ]
)
def test_distinct_multi_answer_options(selected, accepted):
    points = 4
    choices, correct = set(selected), set(accepted)
    if choices == correct:
        expected = points
    elif choices and choices < correct:
        expected = points / 2
    else:
        expected = 0

    # Post-processing returns a tuple for a parsed answer or an empty string.
    prediction = (selected,) if selected else ""
    assert GaokaoBenchMetric._calculate_score(prediction, reference([accepted], points)) == expected


@pytest.mark.parametrize("task", ["multi_mcqs", "seven_option"])
def test_ordered_answers_keep_repeated_letters(task):
    refs = reference(["A", "C", "A"], task=task)
    assert GaokaoBenchMetric._calculate_score(("A", "C", "A"), refs) == 12
    assert GaokaoBenchMetric._calculate_score(("A", "A", "C"), refs) == 4


def test_single_answer_requires_exact_match():
    refs = reference(["A"], task="single_answer_mcq")
    assert GaokaoBenchMetric._calculate_score(("A",), refs) == 4
    assert GaokaoBenchMetric._calculate_score(("AA",), refs) == 0


def test_real_post_processing_and_metric():
    dataset = Gaokao.__new__(Gaokao)
    dataset.subset_name = "2010-2022_Physics_MCQs"
    dataset.evaluation_data = [
        {
            "answer": ["AB"],
            "score": 4
        },
        {
            "answer": ["AB"],
            "score": 6
        },
        {
            "answer": ["AB"],
            "score": 4
        },
        {
            "answer": ["AB"],
            "score": 2
        },
    ]
    responses = [f"【解析】 ... <eoe>\n【答案】: {letters} <eoa>" for letters in ("A A", "A B A", "a b", "A C")]
    assert Gaokao.extract_choice_answer(responses[0], "multi_answers_mcq") == ["AA"]
    predictions = dataset.post_processing(responses)
    assert predictions == [("AA",), ("ABA",), "", ("AC",)]

    metric = GaokaoBenchMetric()
    result = metric(predictions, dataset.references)
    assert metric.last_score_lists == {"Scoring rate": [2, 6, 0, 0]}
    assert result == {"Scoring rate": pytest.approx(8 / 16 * 100)}


@pytest.mark.parametrize("strategy,first_points", [("none", 4), ("leave_one_out", 2)])
def test_multi_reference_aggregation_uses_existing_strategy(strategy, first_points):
    metric = GaokaoBenchMetric(multiref_strategy=strategy)
    predictions = [("BA",), ("A",), ("A", "A")]
    references = [
        [reference(["AB"]), reference(["C"])],
        [reference(["AB"])],
        [reference(["A", "A"], score=3, task="multi_mcqs")],
    ]
    result = metric(predictions, references)
    assert metric.last_score_lists == {"Scoring rate": [first_points, 2, 6]}
    assert result == {"Scoring rate": pytest.approx((first_points + 2 + 6) / 14 * 100)}
