# pylint: disable=missing-function-docstring

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from cosine_similarity_top_k.cosine_similarity_top_k import cosine_similarity_top_k


@pytest.mark.parametrize("n_items, top_k", [(5000, 66), (10000, 100)])
def test_cosine_similarity_top_k_big(n_items, top_k):
    embedding_dim = 50
    max_memory_to_use = int(0.1e9)  # force chinking by taking small amount of memory ~100MB
    embeddings = np.random.randn(n_items, embedding_dim)
    expected = cosine_similarity(embeddings)
    top_k_j = np.argpartition(expected, -top_k)[:, -top_k:]
    values = np.take_along_axis(expected, top_k_j, axis=1).flatten()
    indices = top_k_j.flatten()
    indptr = np.arange(0, top_k * (1 + n_items), top_k)
    expected = csr_matrix((values, indices, indptr), shape=(n_items, n_items))
    calculated = cosine_similarity_top_k(embeddings, top_k, max_memory_to_use)
    np.testing.assert_array_almost_equal(calculated.toarray(), expected.toarray())


@pytest.mark.parametrize("n_items, top_k", [(5000, -66), (10000, -100)])
def test_cosine_similarity_negative_top_k_big(n_items, top_k):
    embedding_dim = 50
    max_memory_to_use = int(0.1e9)  # force chinking by taking small amount of memory ~100MB
    embeddings = np.random.randn(n_items, embedding_dim)
    expected = cosine_similarity(embeddings)
    top_k_j = np.argpartition(expected, -top_k)[:, :-top_k]
    values = np.take_along_axis(expected, top_k_j, axis=1).flatten()
    indices = top_k_j.flatten()
    indptr = np.arange(0, abs(top_k) * (1 + n_items), abs(top_k))
    expected = csr_matrix((values, indices, indptr), shape=(n_items, n_items))
    calculated = cosine_similarity_top_k(embeddings, top_k, max_memory_to_use)
    np.testing.assert_array_almost_equal(calculated.toarray(), expected.toarray())


@pytest.mark.parametrize("n_items", [10, 51, 100, 732, 1000])
@pytest.mark.parametrize("embedding_dim", [10, 33, 66, 100])
def test_cosine_similarity_full(n_items, embedding_dim):
    embeddings = np.random.randn(n_items, embedding_dim)

    top_k = n_items
    expected = cosine_similarity(embeddings)
    calculated = cosine_similarity_top_k(embeddings, top_k)
    np.testing.assert_array_almost_equal(calculated.toarray(), expected)

    top_k = None
    expected = cosine_similarity(embeddings)
    calculated = cosine_similarity_top_k(embeddings, top_k)
    np.testing.assert_array_almost_equal(calculated.toarray(), expected)


@pytest.mark.parametrize("n_items, top_k", [(10, 4), (51, 10), (100, 15), (732, 50), (1000, 77)])
@pytest.mark.parametrize("embedding_dim", [10, 33, 66, 100])
def test_cosine_similarity_top_k(n_items, top_k, embedding_dim):
    embeddings = np.random.randn(n_items, embedding_dim)
    expected = cosine_similarity(embeddings)
    top_k_j = np.argpartition(expected, -top_k)[:, -top_k:]
    values = np.take_along_axis(expected, top_k_j, axis=1).flatten()
    indices = top_k_j.flatten()
    indptr = np.arange(0, top_k * (1 + n_items), top_k)
    expected = csr_matrix((values, indices, indptr), shape=(n_items, n_items))
    calculated = cosine_similarity_top_k(embeddings, top_k)
    np.testing.assert_array_almost_equal(calculated.toarray(), expected.toarray())


@pytest.mark.parametrize("float_type", ["float32", "float64"])
@pytest.mark.parametrize("number_type", ["int8", "int16", "int32", "int64", "float32", "float64"])
def test_cosine_similarity_top_k_manual(number_type, float_type):
    embeddings = np.array(
        [
            [1, 3, 5, 2, -1, 10],
            [11, 1, 7, 2, -5, 1],
            [-4, 6, 2, 9, 7, 0],
            [23, 65, -34, 1, 44, 56],
        ]
    ).astype(number_type)
    top_k = 3
    expected = np.array(
        [
            [1.0, 0.40536558, 0, 0.45644864],
            [0.40536558, 1.0, 0, -0.05518936],
            [0.21689401, 0, 1.0, 0.38271049],
            [0.45644864, 0, 0.38271049, 1.0],
        ]
    ).astype(float_type)

    calculated = cosine_similarity_top_k(embeddings, top_k, float_type=float_type)
    assert calculated.dtype == expected.dtype
    np.testing.assert_array_almost_equal(calculated.toarray(), expected)


@pytest.mark.parametrize("number_type", ["uint8", "uint16", "uint32", "uint64"])
def test_cosine_similarity_top_k_manual_unit(number_type):
    embeddings = np.array(
        [
            [1, 3, 5, 2, 1, 10],
            [11, 1, 7, 2, 5, 1],
            [4, 6, 2, 9, 7, 0],
            [23, 65, 34, 1, 44, 56],
        ]
    ).astype(number_type)
    top_k = 3
    expected = np.array(
        [
            [1.0, 0.40536558, 0, 0.80160769],
            [0, 1.0, 0.60510586, 0.56131614],
            [0, 0.60510586, 1.0, 0.60659962],
            [0.80160769, 0, 0.60659962, 1.0],
        ]
    )

    calculated = cosine_similarity_top_k(embeddings, top_k)
    np.testing.assert_array_almost_equal(calculated.toarray(), expected)


@pytest.mark.parametrize("float_type", ["float32", "float64"])
@pytest.mark.parametrize("number_type", ["int8", "int16", "int32", "int64", "float32", "float64"])
def test_cosine_similarity_negative_top_k_manual(number_type, float_type):
    embeddings = np.array(
        [
            [1, 3, 5, 2, -1, 10],
            [11, 1, 7, 2, -5, 1],
            [-4, 6, 2, 9, 7, 0],
            [23, 65, -34, 1, 44, 56],
        ]
    ).astype(number_type)
    top_k = -2
    expected = np.array(
        [
            [0, 0.40536558, 0.21689401, 0],
            [0, 0, -0.21204564, -0.05518936],
            [0.21689401, -0.21204564, 0, 0],
            [0, -0.05518936, 0.38271049, 0],
        ]
    ).astype(float_type)
    calculated = cosine_similarity_top_k(embeddings, top_k, float_type=float_type)
    assert calculated.dtype == expected.dtype
    np.testing.assert_array_almost_equal(calculated.toarray(), expected)
