"""Turn a discovered binding into the submission the clustering driver runs.

Discovery finds which of a team's functions take photos and return one label
per photo. The driver wants an object with ``cluster(images, *, seed)``. This
is the short piece between them, and it lives here because the protocol it
writes to is Week 2's.

Their functions are called exactly as the search called them, because the
search is what proved the chain works. Nothing is repaired: a function that
raises raises, and the driver records it against the scenario, which is how a
bug in their code stays visible as theirs.
"""

from __future__ import annotations

from typing import Any, List, Sequence

__all__ = ["DiscoveredClustering", "build"]


class DiscoveredClustering:
    """A team's own functions, wearing the interface the driver expects."""

    def __init__(self, chain: Sequence[Any]) -> None:
        self._chain = list(chain)

    def cluster(self, images: Sequence[Any], *, seed: int = 0) -> List[Any]:
        """Photos in, one label per photo out.

        ``seed`` is accepted and not passed on. Their functions were bound by
        calling them with photos alone, which is the signature the whole
        corpus wrote; a team who takes a seed is welcome to, and the benchmark
        already measures how much their answer moves between draws.
        """

        from .roles import _run

        labels = _run(self._chain, list(images))
        return list(labels)


def build(submission: Any) -> DiscoveredClustering:
    if not getattr(submission, "ready", False):
        raise RuntimeError("This repository did not resolve, so there is nothing to run.")
    return DiscoveredClustering(submission.chain)
