import numpy as np
import pytest
from scipy.sparse import csr_matrix, rand as srand
from sklearn.metrics.pairwise import cosine_similarity

from chunkdot.cosine_similarity_top_k import cosine_similarity_top_k


def get_top_k(matrix, top_k):
    n_rows = len(matrix)
    n_cols = matrix.shape[1]
    top_k_j = np.argpartition(matrix, -top_k)
    if top_k > 0:
        top_k_j = top_k_j[:, -top_k:]
    else:
        top_k_j = top_k_j[:, :-top_k]
    values = np.take_along_axis(matrix, top_k_j, axis=1).flatten()
    indices = top_k_j.flatten()
    indptr = np.arange(0, abs(top_k) * (1 + n_rows), abs(top_k))
    return csr_matrix((values, indices, indptr), shape=(n_rows, n_cols))


@pytest.mark.parametrize("n_items, top_k", [(5000, 66), (10000, 100)])
def test_cosine_similarity_top_k_big(n_items, top_k):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    np.random.seed(seed=21)
    embeddings = np.random.randn(n_items, embedding_dim)
    expected = cosine_similarity(embeddings)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(embeddings, top_k=top_k, max_memory=max_memory)
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items, top_k", [(5000, 66), (10000, 100)])
def test_cosine_similarity_top_k_big_with_right(n_items, top_k):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    np.random.seed(seed=21)
    embeddings = np.random.randn(n_items, embedding_dim)
    idx = np.random.randint(n_items, size=n_items - 1)
    embeddings_right = embeddings[idx]
    expected = cosine_similarity(embeddings, embeddings_right)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(
        embeddings, embeddings_right=embeddings_right, top_k=top_k, max_memory=max_memory
    )
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items, top_k", [(5000, 66), (10000, 100)])
@pytest.mark.parametrize("density", [0.25, 0.1, 0.01])
@pytest.mark.parametrize("sparse_format", ["csr", "csc", "coo"])
def test_cosine_similarity_top_k_big_sparse(n_items, top_k, density, sparse_format):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    embeddings = srand(
        n_items, embedding_dim, density=density, format=sparse_format, random_state=21
    )
    expected = cosine_similarity(embeddings)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(embeddings, top_k=top_k, max_memory=max_memory)
    # There might be elements with the same similarity and it might be that in the
    # numpy implementation of get top K items returned are different than in
    # the numba implementation.
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items, top_k", [(5000, 66), (10000, 100)])
@pytest.mark.parametrize("density", [0.25, 0.1, 0.01])
@pytest.mark.parametrize("sparse_format", ["csr", "csc", "coo"])
def test_cosine_similarity_top_k_big_sparse_with_right(n_items, top_k, density, sparse_format):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    embeddings = srand(
        n_items, embedding_dim, density=density, format=sparse_format, random_state=21
    )
    idx = np.random.randint(n_items, size=n_items - 1)
    if embeddings.getformat() == "coo":
        embeddings_right = embeddings.tocsr()[idx].tocoo()
    else:
        embeddings_right = embeddings[idx]
    expected = cosine_similarity(embeddings, embeddings_right)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(
        embeddings, embeddings_right=embeddings_right, top_k=top_k, max_memory=max_memory
    )
    # There might be elements with the same similarity and it might be that in the
    # numpy implementation of get top K items returned are different than in
    # the numba implementation.
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items, top_k", [(5000, -66), (10000, -100)])
def test_cosine_similarity_negative_top_k_big(n_items, top_k):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    np.random.seed(seed=21)
    embeddings = np.random.randn(n_items, embedding_dim)
    expected = cosine_similarity(embeddings)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(embeddings, top_k=top_k, max_memory=max_memory)
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(calculated.toarray(), expected.toarray())


@pytest.mark.parametrize("n_items, top_k", [(5000, -66), (10000, -100)])
def test_cosine_similarity_negative_top_k_big_with_right(n_items, top_k):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    np.random.seed(seed=21)
    embeddings = np.random.randn(n_items, embedding_dim)
    idx = np.random.randint(n_items, size=n_items - 1)
    embeddings_right = embeddings[idx]
    expected = cosine_similarity(embeddings, embeddings_right)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(
        embeddings, embeddings_right=embeddings_right, top_k=top_k, max_memory=max_memory
    )
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items, top_k", [(5000, -66), (10000, -100)])
@pytest.mark.parametrize("density", [0.25, 0.1, 0.01])
@pytest.mark.parametrize("sparse_format", ["csr", "csc", "coo"])
def test_cosine_similarity_negative_top_k_big_sparse(n_items, top_k, density, sparse_format):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    embeddings = srand(
        n_items, embedding_dim, density=density, format=sparse_format, random_state=21
    )
    expected = cosine_similarity(embeddings)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(embeddings, top_k=top_k, max_memory=max_memory)
    # There might be elements with the same similarity and it might be that in the
    # numpy implementation of get top K items returned are different than in
    # the numba implementation.
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items, top_k", [(5000, -66), (10000, -100)])
@pytest.mark.parametrize("density", [0.25, 0.1, 0.01])
@pytest.mark.parametrize("sparse_format", ["csr", "csc", "coo"])
def test_cosine_similarity_negative_top_k_big_sparse_with_right(
    n_items, top_k, density, sparse_format
):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    embeddings = srand(
        n_items, embedding_dim, density=density, format=sparse_format, random_state=21
    )
    idx = np.random.randint(n_items, size=n_items - 1)
    if embeddings.getformat() == "coo":
        embeddings_right = embeddings.tocsr()[idx].tocoo()
    else:
        embeddings_right = embeddings[idx]
    expected = cosine_similarity(embeddings, embeddings_right)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(
        embeddings, embeddings_right=embeddings_right, top_k=top_k, max_memory=max_memory
    )
    # There might be elements with the same similarity and it might be that in the
    # numpy implementation of get top K items returned are different than in
    # the numba implementation.
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items", [10, 1000])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
def test_cosine_similarity_error(n_items, as_csr_sparse):
    np.random.seed(seed=21)
    embeddings = np.random.randn(n_items, 100)
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)

    with pytest.raises(ValueError):
        cosine_similarity_top_k(embeddings, top_k=n_items)

    with pytest.raises(ValueError):
        cosine_similarity_top_k(embeddings, embeddings_right=embeddings, top_k=n_items)

    # make max memory small such that not even one chunk can be processed
    max_memory = 16 * n_items + 8 * n_items * (2 * n_items + 1) - 10
    with pytest.raises(ValueError):
        cosine_similarity_top_k(embeddings, top_k=n_items, max_memory=max_memory)


@pytest.mark.parametrize("as_csr_sparse", [False, True])
def test_cosine_similarity_mix_type_error(as_csr_sparse):
    n_items = 10
    top_k = 2
    np.random.seed(seed=21)
    embeddings = np.random.randn(n_items, 100)
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
        embeddings_right = embeddings.todense()
    else:
        embeddings_right = csr_matrix(embeddings)

    with pytest.raises(TypeError):
        cosine_similarity_top_k(embeddings, embeddings_right=embeddings_right, top_k=top_k)


@pytest.mark.parametrize("n_items, top_k", [(10, 4), (51, 10), (100, 15), (732, 50), (1000, 77)])
@pytest.mark.parametrize("embedding_dim", [10, 33, 66, 100])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
def test_cosine_similarity_top_k(n_items, top_k, embedding_dim, as_csr_sparse):
    np.random.seed(seed=21)
    embeddings = np.random.randn(n_items, embedding_dim)
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
    expected = cosine_similarity(embeddings)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(embeddings, top_k=top_k)
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items, top_k", [(10, 4), (51, 10), (100, 15), (732, 50), (1000, 77)])
@pytest.mark.parametrize("embedding_dim", [10, 33, 66, 100])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
def test_cosine_similarity_top_k_with_right(n_items, top_k, embedding_dim, as_csr_sparse):
    np.random.seed(seed=21)
    embeddings = np.random.randn(n_items, embedding_dim)
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
    idx = np.random.randint(n_items, size=n_items - 1)
    embeddings_right = embeddings[idx]
    expected = cosine_similarity(embeddings, embeddings_right)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(embeddings, embeddings_right=embeddings_right, top_k=top_k)
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("input_type", ["int8", "int16", "int32", "int64", "float32", "float64"])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
@pytest.mark.parametrize("use_embeddings_right", [False, True])
def test_cosine_similarity_top_k_manual(input_type, as_csr_sparse, use_embeddings_right):
    embeddings = np.array(
        [
            [1, 3, 5, 2, -1, 10],
            [11, 1, 7, 2, -5, 1],
            [-4, 6, 2, 9, 7, 0],
            [23, 65, -34, 1, 44, 56],
        ]
    ).astype(input_type)
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
    top_k = 3
    if use_embeddings_right:
        expected_type = cosine_similarity(embeddings, embeddings).dtype
    else:
        expected_type = cosine_similarity(embeddings).dtype
    expected = np.array(
        [
            [1.0, 0.40536558, 0, 0.45644864],
            [0.40536558, 1.0, 0, -0.05518936],
            [0.21689401, 0, 1.0, 0.38271049],
            [0.45644864, 0, 0.38271049, 1.0],
        ]
    ).astype(expected_type)
    if use_embeddings_right:
        calculated = cosine_similarity_top_k(embeddings, embeddings_right=embeddings, top_k=top_k)
    else:
        calculated = cosine_similarity_top_k(embeddings, top_k=top_k)
    assert calculated.dtype == expected.dtype
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(calculated.toarray(), expected)


@pytest.mark.parametrize("input_type", ["uint8", "uint16", "uint32", "uint64"])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
@pytest.mark.parametrize("use_embeddings_right", [False, True])
def test_cosine_similarity_top_k_manual_unit(input_type, as_csr_sparse, use_embeddings_right):
    embeddings = np.array(
        [
            [1, 3, 5, 2, 1, 10],
            [11, 1, 7, 2, 5, 1],
            [4, 6, 2, 9, 7, 0],
            [23, 65, 34, 1, 44, 56],
        ]
    ).astype(input_type)
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
    top_k = 3
    if use_embeddings_right:
        expected_type = cosine_similarity(embeddings, embeddings).dtype
    else:
        expected_type = cosine_similarity(embeddings).dtype
    expected = np.array(
        [
            [1.0, 0.40536558, 0, 0.80160769],
            [0, 1.0, 0.60510586, 0.56131614],
            [0, 0.60510586, 1.0, 0.60659962],
            [0.80160769, 0, 0.60659962, 1.0],
        ]
    ).astype(expected_type)
    if use_embeddings_right:
        calculated = cosine_similarity_top_k(embeddings, embeddings_right=embeddings, top_k=top_k)
    else:
        calculated = cosine_similarity_top_k(embeddings, top_k=top_k)
    assert calculated.dtype == expected.dtype
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(calculated.toarray(), expected)


@pytest.mark.parametrize("input_type", ["int8", "int16", "int32", "int64", "float32", "float64"])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
@pytest.mark.parametrize("use_embeddings_right", [False, True])
def test_cosine_similarity_negative_top_k_manual(input_type, as_csr_sparse, use_embeddings_right):
    embeddings = np.array(
        [
            [1, 3, 5, 2, -1, 10],
            [11, 1, 7, 2, -5, 1],
            [-4, 6, 2, 9, 7, 0],
            [23, 65, -34, 1, 44, 56],
        ]
    ).astype(input_type)
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
    top_k = -2
    if use_embeddings_right:
        expected_type = cosine_similarity(embeddings, embeddings).dtype
    else:
        expected_type = cosine_similarity(embeddings).dtype
    expected = np.array(
        [
            [0, 0.40536558, 0.21689401, 0],
            [0, 0, -0.21204564, -0.05518936],
            [0.21689401, -0.21204564, 0, 0],
            [0, -0.05518936, 0.38271049, 0],
        ]
    ).astype(expected_type)
    if use_embeddings_right:
        calculated = cosine_similarity_top_k(embeddings, embeddings_right=embeddings, top_k=top_k)
    else:
        calculated = cosine_similarity_top_k(embeddings, top_k=top_k)
    assert calculated.dtype == expected.dtype
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(calculated.toarray(), expected)


@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
@pytest.mark.parametrize("use_embeddings_right", [False, True])
def test_cosine_similarity_top_k_zero_rows(top_k, as_csr_sparse, use_embeddings_right):
    embeddings = np.array([[0, 0, 0], [34, 22, 11], [0, 0, 0], [11, 21, 34]])
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
    if use_embeddings_right:
        idx = np.random.randint(4, size=4)
        embeddings_right = embeddings[idx]
        expected = cosine_similarity(embeddings, embeddings_right)
    else:
        expected = cosine_similarity(embeddings)
    expected = get_top_k(expected, top_k)
    if use_embeddings_right:
        calculated = cosine_similarity_top_k(
            embeddings, embeddings_right=embeddings_right, top_k=top_k
        )
    else:
        calculated = cosine_similarity_top_k(embeddings, top_k=top_k)
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("top_k", [-1, -2, -3])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
@pytest.mark.parametrize("use_embeddings_right", [False, True])
def test_cosine_similarity_negative_top_k_zero_rows(top_k, as_csr_sparse, use_embeddings_right):
    embeddings = np.array([[0, 0, 0], [34, 22, 11], [0, 0, 0], [11, 21, 34]])
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
    if use_embeddings_right:
        idx = np.random.randint(4, size=4)
        embeddings_right = embeddings[idx]
        expected = cosine_similarity(embeddings, embeddings_right)
    else:
        expected = cosine_similarity(embeddings)
    expected = get_top_k(expected, top_k)
    if use_embeddings_right:
        calculated = cosine_similarity_top_k(
            embeddings, embeddings_right=embeddings_right, top_k=top_k
        )
    else:
        calculated = cosine_similarity_top_k(embeddings, top_k=top_k)
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(np.sort(calculated.data), np.sort(expected.data))
    assert len(calculated.indices) == len(expected.indices)
    np.testing.assert_array_almost_equal(calculated.indptr, expected.indptr)


@pytest.mark.parametrize("n_items, top_k, show_progress", [(5000, -15, True), (5000, -15, False)])
@pytest.mark.parametrize("as_csr_sparse", [False, True])
def test_cosine_similarity_with_progress_bar(n_items, top_k, show_progress, as_csr_sparse):
    embedding_dim = 50
    max_memory = int(50e6)  # force chunking by taking small amount of memory ~50MB
    embeddings = np.random.randn(n_items, embedding_dim)
    if as_csr_sparse:
        embeddings = csr_matrix(embeddings)
    expected = cosine_similarity(embeddings)
    expected = get_top_k(expected, top_k)
    calculated = cosine_similarity_top_k(
        embeddings, top_k=top_k, max_memory=max_memory, show_progress=show_progress
    )
    assert calculated.shape == expected.shape
    np.testing.assert_array_almost_equal(calculated.toarray(), expected.toarray())
