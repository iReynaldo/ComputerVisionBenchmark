"""CogBench plugin objects for the two Week 2 vision tracks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .datasets import (
    CacheStatus,
    cache_status,
    clustering_scenarios,
    load_manifest,
    recognition_scenarios,
)
from .drivers import (
    recognition_expected,
    run_clustering_scenario,
    run_recognition_scenario,
)
from .metrics import score_clustering, score_recognition


class RecognitionBenchmark:
    """Load, execute, and score known/unknown recognition scenarios."""

    benchmark_id = "vision-recognition"
    benchmark_version = 2
    contract_version = "cogworks.submissions.v2"
    plugin_version = "0.1.0"
    dataset_version = "celeba-manifests-v1"
    scorer_version = "recognition-v1"
    primary_metric = "recognition_score"

    metric_labels = {
        "recognition_score": "Recognition score",
        "known_identification": "Known identification",
        "unknown_rejection_recall": "Unknown rejection recall",
        "post_enrollment_accuracy": "Post-enrollment accuracy",
        "unknown_lifecycle": "Unknown lifecycle",
    }

    #: What each metric measures, in the course's own vocabulary, and which
    #: part of the capstone it comes from.
    #:
    #: A number a student cannot trace back to something they were taught is a
    #: black box, and a black box teaches nothing. The words here are CogWeb's
    #: -- "descriptor vector", "cosine distance", "cutoff" -- rather than ours.
    #: Source: docs/cogweb/pages/Video/FacialRecognition.md.
    #: Filled by score(). Present before the first run so a caller reading it
    #: early gets an empty list rather than an AttributeError.
    last_diagnostics: list = []

    metric_help = {
        "recognition_score": (
            "The mean of known identification and the unknown lifecycle below. "
            "Both halves count equally because the capstone is not only "
            "recognizing a face it has seen: saying \"I do not know this person\" "
            "and then learning them is the other half of the assignment. This is "
            "the leaderboard number."
        ),
        "known_identification": (
            "Of the people already in the database, how often the right name comes "
            "back. Averaged per person rather than per image, so someone with many "
            "photos cannot carry the score for someone with few. This is the "
            "cosine distance between descriptor vectors doing its job."
        ),
        "unknown_rejection_recall": (
            "How often a face that is not in the database is correctly called "
            "unknown rather than matched to the nearest name. This is the cutoff "
            "the course asks you to choose: too generous and every stranger "
            "becomes somebody, too strict and nobody is ever recognized."
        ),
        "post_enrollment_accuracy": (
            "After a stranger is added to the database, how often they are then "
            "identified correctly. A system that rejects unknowns but cannot learn "
            "them has only solved half the problem."
        ),
        "unknown_lifecycle": (
            "The mean of unknown rejection and post-enrollment accuracy: the whole "
            "arc of meeting a stranger, saying so, and knowing them next time."
        ),
    }

    def load_cases(self, tier: str, cache_root: Optional[Path] = None) -> Sequence[Any]:
        """Load the small test or larger public-evaluation cases."""

        return recognition_scenarios(_public_manifest(tier), cache_root)

    def run(
        self, factory: Any, model: Any, cases: Sequence[Any]
    ) -> List[Mapping[str, Sequence[Optional[str]]]]:
        """Execute every case through a fresh application adapter."""

        return [run_recognition_scenario(factory, model, case) for case in cases]

    def score(
        self,
        outputs: Sequence[Mapping[str, Sequence[Optional[str]]]],
        cases: Sequence[Any],
    ) -> Dict[str, float]:
        """Compare recognition outputs with trusted lifecycle labels.

        `score_recognition` returns its diagnostics alongside the numbers;
        they are lifted onto `last_diagnostics` here, matching Weeks 1 and 3,
        so the returned mapping stays metric-name to float and the runner can
        float() every value in it without special-casing a key.
        """

        scores = dict(score_recognition(outputs, [recognition_expected(case) for case in cases]))
        self.last_diagnostics = list(scores.pop("_diagnostics", []))
        return scores

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        """Report whether the selected public data tier is cached and valid."""

        return cache_status(_public_manifest(tier), cache_root)


class ClusteringBenchmark:
    """Load, execute, and score fixed-seed Whispers scenarios."""

    benchmark_id = "vision-clustering"
    benchmark_version = 2
    contract_version = "cogworks.submissions.v2"
    plugin_version = "0.1.0"
    dataset_version = "celeba-manifests-v1"
    scorer_version = "clustering-v1"
    primary_metric = "clustering_pairwise_f1"

    metric_labels = {
        "clustering_pairwise_f1": "Pairwise F1",
        "adjusted_rand_index": "Adjusted Rand index",
    }

    #: See RecognitionBenchmark.metric_help. Source for the vocabulary:
    #: docs/cogweb/pages/Video/Whispers.md.
    metric_help = {
        "clustering_pairwise_f1": (
            "Over every pair of photos, how often two photos of the same person "
            "end up in the same group and two photos of different people end up "
            "apart. Cluster numbers themselves do not matter -- calling a group "
            "3 instead of 1 changes nothing -- which is why the score is built "
            "from pairs. This is the leaderboard number."
        ),
        "adjusted_rand_index": (
            "The same agreement, corrected for how much a random grouping would "
            "score by luck. Reported next to pairwise F1 because a run that "
            "puts every photo in one giant cluster can look respectable on F1 "
            "and lands near zero here."
        ),
    }

    def load_cases(self, tier: str, cache_root: Optional[Path] = None) -> Sequence[Any]:
        """Load the small test or larger public-evaluation cases."""

        return clustering_scenarios(_public_manifest(tier), cache_root)

    def run(self, factory: Any, model: Any, cases: Sequence[Any]) -> List[Sequence[Any]]:
        """Execute every clustering case through a fresh adapter."""

        return [run_clustering_scenario(factory, model, case) for case in cases]

    def score(self, outputs: Sequence[Sequence[Any]], cases: Sequence[Any]) -> Dict[str, float]:
        """Compute label-invariant metrics against trusted partitions."""

        return score_clustering(outputs, [case.expected_labels for case in cases])

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        """Report whether the selected public data tier is cached and valid."""

        return cache_status(_public_manifest(tier), cache_root)


def _public_manifest(tier: str) -> Mapping[str, Any]:
    if tier not in ("test", "evaluation"):
        raise ValueError("Public tier must be 'test' or 'evaluation'.")
    return load_manifest(tier)
