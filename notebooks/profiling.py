import timeit
import tracemalloc
import numpy as np
import scipy


def get_memory(
    cosine_sim_function,
    *,
    n_items: int,
    embedding_dim: int,
    float_type="float64",
    function_kwargs=None,
):
    """Get the memory used by the cosine_sim_function."""
    embeddings = np.random.randn(int(n_items), int(embedding_dim)).astype(float_type)
    tracemalloc.clear_traces()
    tracemalloc.start()
    if function_kwargs is not None:
        similarity = cosine_sim_function(embeddings, **function_kwargs)
    else:
        similarity = cosine_sim_function(embeddings)
    _, max_size = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if isinstance(similarity, scipy.sparse.csr_matrix):
        matrix_bytes = similarity.data.nbytes + similarity.indptr.nbytes + similarity.indices.nbytes
    else:
        matrix_bytes = similarity.nbytes
    return max_size, matrix_bytes


def get_time(
    cosine_sim_function,
    *,
    n_items,
    embedding_dim,
    float_type="float64",
    n_iterations=10,
    function_kwargs=None,
):
    """Get the execution time of the cosine_sim_function."""
    embeddings = np.random.randn(n_items, embedding_dim).astype(float_type)

    def _similarity():
        if function_kwargs is not None:
            cosine_sim_function(embeddings, **function_kwargs)
        else:
            cosine_sim_function(embeddings)

    result = timeit.timeit(stmt="_similarity()", globals=locals(), number=n_iterations)
    return result / n_iterations
