from __future__ import annotations

from benchmark_adapter import create_clustering_adapter, create_recognition_adapter


def test_recognition_factory_exposes_contract(fake_model):
    adapter = create_recognition_adapter(fake_model)
    assert callable(adapter.enroll)
    assert callable(adapter.recognize)


def test_clustering_factory_exposes_contract(fake_model):
    adapter = create_clustering_adapter(fake_model)
    assert callable(adapter.cluster)
