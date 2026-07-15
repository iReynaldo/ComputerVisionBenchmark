from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np


class FaceRecognitionApp:
    """Optional broad application shape; teams may replace it entirely."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def detect(self, image: np.ndarray) -> Any:
        return self.model.detect(image)

    def compute_descriptors(self, image: np.ndarray, boxes: Any) -> np.ndarray:
        return self.model.compute_descriptors(image, boxes)

    def process_image(self, image_path: str) -> Mapping[str, Any]:
        raise NotImplementedError("Implement your recognition pipeline.")

    def recognize_face(self, descriptor: np.ndarray) -> Tuple[str, float]:
        raise NotImplementedError("Implement your match rule.")

    def add_image_to_database(self, image_path: str, name: str) -> bool:
        raise NotImplementedError("Implement enrollment for your database design.")

    def add_descriptors_to_database(self, descriptors: Sequence[np.ndarray], name: str) -> None:
        raise NotImplementedError("Implement enrollment for your database design.")

    def cluster_images(self, images: Sequence[np.ndarray]) -> Sequence[Union[str, int]]:
        raise NotImplementedError("Implement Whispers clustering.")

    def save_database(self, filepath: Union[str, Path]) -> None:
        raise NotImplementedError("Persistence is optional application behavior.")

    def load_database(self, filepath: Union[str, Path]) -> None:
        raise NotImplementedError("Persistence is optional application behavior.")

    def get_database_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError("Expose data only if it is useful to your application.")
