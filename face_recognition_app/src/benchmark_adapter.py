from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
from app import FaceRecognitionApp


class RecognitionAdapter:
    """Translate the benchmark contract into your application's own API."""

    def __init__(self, model: Any) -> None:
        self.app = FaceRecognitionApp(model)

    def enroll(self, person_id: str, images: Sequence[np.ndarray]) -> None:
        raise NotImplementedError("Map enrollment into your application here.")

    def recognize(self, images: Sequence[np.ndarray]) -> Sequence[Optional[str]]:
        raise NotImplementedError("Map recognition into your application here.")


class ClusteringAdapter:
    """Translate the clustering contract into your application's own API."""

    def __init__(self, model: Any) -> None:
        self.app = FaceRecognitionApp(model)

    def cluster(self, images: Sequence[np.ndarray], *, seed: int) -> Sequence[Union[str, int]]:
        raise NotImplementedError("Map deterministic clustering into your application here.")


def create_recognition_adapter(model: Any) -> RecognitionAdapter:
    return RecognitionAdapter(model)


def create_clustering_adapter(model: Any) -> ClusteringAdapter:
    return ClusteringAdapter(model)
