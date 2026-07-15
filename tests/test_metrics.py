from __future__ import annotations

import pytest

from facial_recognition_benchmark.metrics import (
    adjusted_rand_index,
    pairwise_f1,
    score_clustering,
    score_recognition,
)

EXPECTED_RECOGNITION = [
    {
        "known": ["a", "a", "b"],
        "unknown_before": [None, None],
        "post_enrollment": ["new", "new"],
    }
]


def test_perfect_recognition_scores_every_behavior():
    score = score_recognition(EXPECTED_RECOGNITION, EXPECTED_RECOGNITION)
    assert score["known_identification"] == 1.0
    assert score["unknown_lifecycle"] == 1.0
    assert score["recognition_score"] == 1.0


def test_always_unknown_does_not_get_misleading_recognition_score():
    output = [{"known": [None] * 3, "unknown_before": [None] * 2, "post_enrollment": [None] * 2}]
    score = score_recognition(output, EXPECTED_RECOGNITION)
    assert score["unknown_rejection_recall"] == 1.0
    assert score["post_enrollment_accuracy"] == 0.0
    assert score["unknown_lifecycle"] == 0.5
    assert score["recognition_score"] == 0.25


def test_missing_or_malformed_recognition_is_zero_not_renormalized():
    score = score_recognition([{"known": []}], EXPECTED_RECOGNITION)
    assert set(score.values()) == {0.0}


def test_cluster_label_names_do_not_matter():
    expected = ["first", "first", "second", "second"]
    actual = [9, 9, 2, 2]
    assert pairwise_f1(actual, expected) == 1.0
    assert adjusted_rand_index(actual, expected) == 1.0


@pytest.mark.parametrize("actual", [[0, 1, 2, 3], [0, 0, 0, 0]])
def test_degenerate_clusterings_are_not_perfect(actual):
    score = score_clustering([actual], [[0, 0, 1, 1]])
    assert score["clustering_pairwise_f1"] < 1.0


def test_missing_clustering_case_scores_zero():
    assert score_clustering([], [[0, 0]])["clustering_pairwise_f1"] == 0.0
