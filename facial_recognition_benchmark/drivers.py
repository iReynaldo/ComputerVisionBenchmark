"""Benchmark-owned execution drivers for student application behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .adapters import AdapterContractError, adapt_clustering, adapt_recognition, instantiate
from .contracts import ClusterId, Image, PersonId


@dataclass(frozen=True)
class RecognitionIdentity:
    """Enrollment and held-out images for one known identity."""

    person_id: PersonId
    enrollment: Sequence[Image]
    queries: Sequence[Image]


@dataclass(frozen=True)
class RecognitionScenario:
    """One complete known-to-unknown recognition lifecycle.

    The same fresh adapter is retained across initial enrollment, known
    queries, unknown rejection, enrollment of that unknown identity, and
    held-out re-identification.
    """

    known: Sequence[RecognitionIdentity]
    unknown_person_id: PersonId
    unknown_queries: Sequence[Image]
    unknown_enrollment: Sequence[Image]
    post_enrollment_queries: Sequence[Image]


@dataclass(frozen=True)
class ClusteringScenario:
    """One fixed image partition and random seed for Whispers evaluation.

    ``scored`` marks whether this case contributes to the metrics. The seed
    sweep re-runs the same images under other seeds to report how far a
    randomized clusterer's answer moves; those repeats are diagnostic and
    must not change the number a team publishes.
    """

    images: Sequence[Image]
    expected_labels: Sequence[ClusterId]
    seed: int
    scored: bool = True


RecognitionOutput = Dict[str, List[Optional[PersonId]]]


def run_recognition_scenario(
    factory: Any, model: Any, scenario: RecognitionScenario
) -> RecognitionOutput:
    """Run one stateful recognition lifecycle with a fresh student adapter.

    Parameters
    ----------
    factory
        Raw submission factory or compatible application object.
    model
        Benchmark-owned FaceNet model supplied to the factory.
    scenario
        Enrollment and query sequence to execute.

    Returns
    -------
    dict
        Labels for known, pre-enrollment unknown, and post-enrollment queries.
    """

    adapter = adapt_recognition(instantiate(factory, model))
    for identity in scenario.known:
        adapter.enroll(identity.person_id, identity.enrollment)

    known_images: List[Image] = []
    for identity in scenario.known:
        known_images.extend(identity.queries)
    known = _recognition_labels(adapter.recognize(known_images), len(known_images), "known")

    unknown_before = _recognition_labels(
        adapter.recognize(scenario.unknown_queries),
        len(scenario.unknown_queries),
        "unknown-before-enrollment",
    )
    adapter.enroll(scenario.unknown_person_id, scenario.unknown_enrollment)
    post_enrollment = _recognition_labels(
        adapter.recognize(scenario.post_enrollment_queries),
        len(scenario.post_enrollment_queries),
        "post-enrollment",
    )
    return {
        "known": known,
        "unknown_before": unknown_before,
        "post_enrollment": post_enrollment,
    }


def run_clustering_scenario(
    factory: Any, model: Any, scenario: ClusteringScenario
) -> List[ClusterId]:
    """Run and validate one clustering case with a fresh student adapter.

    Cluster identifiers may be arbitrary strings or integers. Only their
    partition relationships are meaningful to scoring.
    """

    adapter = adapt_clustering(instantiate(factory, model))
    labels = adapter.cluster(scenario.images, seed=scenario.seed)
    if isinstance(labels, (str, bytes)):
        raise AdapterContractError("cluster() must return one label per image, not a string.")
    try:
        output = list(labels)
    except TypeError as error:
        raise AdapterContractError("cluster() must return a sequence of labels.") from error
    if len(output) != len(scenario.images):
        raise AdapterContractError(
            f"cluster() returned {len(output)} labels for {len(scenario.images)} images."
        )
    for index, label in enumerate(output):
        if isinstance(label, np.generic):
            label = label.item()
            output[index] = label
        if isinstance(label, bool) or not isinstance(label, (str, int)):
            raise AdapterContractError(f"cluster() label {index} must be a string or integer.")
    return output


def recognition_expected(scenario: RecognitionScenario) -> RecognitionOutput:
    """Construct trusted labels for a recognition lifecycle scenario."""

    known: List[Optional[PersonId]] = []
    for identity in scenario.known:
        known.extend([identity.person_id] * len(identity.queries))
    return {
        "known": known,
        "unknown_before": [None] * len(scenario.unknown_queries),
        "post_enrollment": [scenario.unknown_person_id] * len(scenario.post_enrollment_queries),
    }


def _recognition_labels(value: Any, expected_count: int, phase: str) -> List[Optional[str]]:
    if isinstance(value, (str, bytes)):
        raise AdapterContractError(
            f"recognize() must return one label per image during {phase}, not a string."
        )
    try:
        output = list(value)
    except TypeError as error:
        raise AdapterContractError(f"recognize() must return a sequence during {phase}.") from error
    if len(output) != expected_count:
        raise AdapterContractError(
            f"recognize() returned {len(output)} labels for {expected_count} images during {phase}."
        )
    normalized: List[Optional[str]] = []
    for index, label in enumerate(output):
        if label is None:
            normalized.append(None)
        elif isinstance(label, str):
            normalized.append(label)
        else:
            raise AdapterContractError(
                f"recognize() label {index} during {phase} must be a string or None."
            )
    return normalized
