import importlib

import pytest

import chunkdot


def test_chunkdot_top_level_imports():
    attrs = ["cosine_similarity_top_k", "chunkdot", "CosineSimilarityTopK"]
    for attr in attrs:
        hasattr(chunkdot, attr)


def test_chunkdot_warning(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    with pytest.warns(UserWarning):
        importlib.reload(chunkdot)
