"""Distribution-level discovery tests for benchmark and reference plugins."""

from __future__ import annotations

import importlib.metadata as metadata
from typing import Dict


def entry_points(group: str) -> Dict[str, metadata.EntryPoint]:
    """Return one entry-point group on Python 3.8 and newer runtimes."""

    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        values = discovered.select(group=group)
    else:
        values = discovered.get(group, ())
    return {point.name: point for point in values}


def test_benchmark_tracks_are_discoverable() -> None:
    """Load both v2 benchmark objects from installed distribution metadata."""

    points = entry_points("cogworks.benchmarks.v2")
    assert set(points) == {"vision-recognition", "vision-clustering"}
    assert points["vision-recognition"].load()().benchmark_version == 2
    assert points["vision-clustering"].load()().benchmark_version == 2


def test_reference_application_tracks_are_discoverable() -> None:
    """Load both golden-application factories without constructing FaceNet."""

    points = entry_points("cogworks.submissions.v2")
    assert set(points) == {"vision-recognition", "vision-clustering"}
    assert callable(points["vision-recognition"].load())
    assert callable(points["vision-clustering"].load())
