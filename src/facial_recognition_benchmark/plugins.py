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
    benchmark_id = "vision-recognition"
    benchmark_version = "2.0.0"
    contract_version = "cogworks.submissions.v2"
    dataset_version = "celeba-manifests-v1"
    scorer_version = "recognition-v1"
    primary_metric = "recognition_score"

    def load_cases(self, tier: str, cache_root: Optional[Path] = None) -> Sequence[Any]:
        return recognition_scenarios(_public_manifest(tier), cache_root)

    def run(
        self, factory: Any, model: Any, cases: Sequence[Any]
    ) -> List[Mapping[str, Sequence[Optional[str]]]]:
        return [run_recognition_scenario(factory, model, case) for case in cases]

    def score(
        self,
        outputs: Sequence[Mapping[str, Sequence[Optional[str]]]],
        cases: Sequence[Any],
    ) -> Dict[str, float]:
        return score_recognition(outputs, [recognition_expected(case) for case in cases])

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        return cache_status(_public_manifest(tier), cache_root)


class ClusteringBenchmark:
    benchmark_id = "vision-clustering"
    benchmark_version = "2.0.0"
    contract_version = "cogworks.submissions.v2"
    dataset_version = "celeba-manifests-v1"
    scorer_version = "clustering-v1"
    primary_metric = "clustering_pairwise_f1"

    def load_cases(self, tier: str, cache_root: Optional[Path] = None) -> Sequence[Any]:
        return clustering_scenarios(_public_manifest(tier), cache_root)

    def run(self, factory: Any, model: Any, cases: Sequence[Any]) -> List[Sequence[Any]]:
        return [run_clustering_scenario(factory, model, case) for case in cases]

    def score(self, outputs: Sequence[Sequence[Any]], cases: Sequence[Any]) -> Dict[str, float]:
        return score_clustering(outputs, [case.expected_labels for case in cases])

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        return cache_status(_public_manifest(tier), cache_root)


def _public_manifest(tier: str) -> Mapping[str, Any]:
    if tier not in ("test", "evaluation"):
        raise ValueError("Public tier must be 'test' or 'evaluation'.")
    return load_manifest(tier)
