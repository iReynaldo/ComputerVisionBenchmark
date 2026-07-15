from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Union

import numpy as np

Image = np.ndarray
PersonId = str
ClusterId = Union[str, int]


class RecognitionAdapter(Protocol):
    def enroll(self, person_id: PersonId, images: Sequence[Image]) -> None: ...

    def recognize(self, images: Sequence[Image]) -> Sequence[Optional[PersonId]]: ...


class ClusteringAdapter(Protocol):
    def cluster(self, images: Sequence[Image], *, seed: int) -> Sequence[ClusterId]: ...


AdapterFactory = Any
