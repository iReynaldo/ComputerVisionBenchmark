"""CogWorks Week 2 behavioral benchmark plugins."""

from .adapters import AdapterContractError, adapt_clustering, adapt_recognition
from .metrics import score_clustering, score_recognition
from .plugins import ClusteringBenchmark, RecognitionBenchmark

__all__ = [
    "AdapterContractError",
    "ClusteringBenchmark",
    "RecognitionBenchmark",
    "adapt_clustering",
    "adapt_recognition",
    "score_clustering",
    "score_recognition",
]

__version__ = "0.1.0"
