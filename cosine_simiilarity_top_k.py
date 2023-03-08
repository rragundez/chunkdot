import logging
import math
import psutil

import numba
import numpy as np
import numba_argpartition
from numba import njit, prange
from scipy.sparse import csr_matrix
import warnings

LOGGER = logging.getLogger(__name__)


def get_memory_available():
    return psutil.virtual_memory().available


def get_chunk_size_per_thread(
    n_items, top_k, max_memory_bytes=None, force_memory=False
):
    M_max = get_memory_available() * 0.9  # use only 90% of the available memory
    M = M_max
    if max_memory_bytes:
        if max_memory_bytes > M_max:
            message = (
                f"Requested maximum memory to use {max_memory_bytes} is bigger than 90% of the "
                f"system's available memory {M_max}."
            )
            if force_memory:
                warnings.warn(message)
                M = max_memory_bytes
            else:
                raise ValueError(message)
        else:
            M = max_memory_bytes

    # get maximum possible matrix of size chunk_size x chunk_size given available memory
    chunk_size = (
        M / (2 * 8) - (2 * n_items * top_k + n_items)
    ) / n_items  # 8 bytes for a float64 or double type
    N = numba.get_num_threads()
    chunk_size_per_thread = math.floor(chunk_size / N)
    LOGGER.debug(f"Memory available: {M_max / 1E9:.2f} GB")
    LOGGER.debug(f"Using memory: {M / 1E9:.2f} GB")
    LOGGER.debug(f"Number of threads: {N}")
    LOGGER.debug(f"Chunk size per thread: {chunk_size_per_thread}")
    return chunk_size_per_thread


@njit(
    parallel=False
)  # . noting to parallaleize in this function will raise a warning if used
def to_sparse(m, top_k):
    n_rows, n_cols = m.shape
    top_k_j = np.argpartition(m, -top_k)[:, -top_k:]
    values = np.take_along_axis(m, top_k_j, axis=1).flatten()
    indices = top_k_j.flatten()
    return values, indices


@njit(parallel=True)
def chunked_dot(x, y, top_k, chunk_size):
    n_rows = len(x)
    n_non_zero = n_rows * top_k
    all_values, all_indices, all_indptr = (
        np.empty(n_non_zero),
        np.empty(n_non_zero),
        np.empty(n_rows + 1),
    )
    all_indptr[0] = 0
    for i in prange(0, math.ceil(len(x) / chunk_size)):
        start_row_i = i * chunk_size
        end_row_i = (i + 1) * chunk_size
        chunk_m = np.dot(x[start_row_i:end_row_i], y)
        values, indices = to_sparse(chunk_m, top_k)
        all_values[start_row_i * top_k : end_row_i * top_k] = values
        all_indices[start_row_i * top_k : end_row_i * top_k] = indices
    # standard CSR form representation
    all_indptr = np.arange(0, top_k * (1 + n_rows), top_k)
    return all_values, all_indices, all_indptr


def warm_up_numba_function():
    x = np.random.randn(10000, 10).astype("double")
    chunked_dot(x, x.T, 3)


def cosine_similarity_top_k(m, top_k=None, max_memory_bytes=None, force_memory=False):
    n_rows = m.shape[0]
    if top_k is None:
        top_k = n_rows
    #     warm_up_numba_function()
    chunk_size_per_thread = get_chunk_size_per_thread(
        n_rows, top_k, max_memory_bytes, force_memory
    )
    values, indices, indptr = chunked_dot(m, m.T, top_k, chunk_size_per_thread)
    return csr_matrix((values, indices, indptr))
