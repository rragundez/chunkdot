import logging
import math
import warnings

import psutil
import numba
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from cosine_similarity_top_k import numba_argpartition  # pylint: disable=unused-import

LOGGER = logging.getLogger(__name__)


def get_memory_available():
    """Get the available memory to the OS.

    The memory that can be given instantly to processes without the system going into swap. This
    is calculated by summing different memory values depending on the platform and it is supposed
    to be used to monitor actual memory usage in a cross platform fashion.

    Note that if called inside a running Python process this might not reflect 1 to 1 the
    available memory on that process, as memory could have been used in the Python process
    and release for further use in the same Python process but not back to the OS even if it has
    been garbage collected.
    """
    return psutil.virtual_memory().available


def get_chunk_size_per_thread(n_items, top_k, max_memory_to_use=None, force_memory=False):
    """Calculate the number of rows that can be calculated.

    Given the number of items, the amount of similar items requested and the limit for the memory
    to consume, calculate the maximum number of rows possible on each chunk to be used in the
    matrix multiplication of the parallel processes.
    """
    # For safety use only 95% of the available memory
    max_memory = get_memory_available() * 0.95
    memory_to_use = max_memory
    if max_memory_to_use:
        if max_memory_to_use > max_memory:
            message = (
                f"Requested memory to use {max_memory_to_use / 1E9:.2f} is bigger than 95% of the "
                f"system's available memory {max_memory / 1E9:.2f}."
            )
            if force_memory:
                warnings.warn(message)
                memory_to_use = max_memory_to_use
            else:
                raise ValueError(message)
        else:
            memory_to_use = max_memory_to_use

    chunk_size = (memory_to_use / (2 * 8) - (2 * n_items * top_k + n_items)) / n_items
    n_of_threads = numba.get_num_threads()
    chunk_size_per_thread = math.floor(chunk_size / n_of_threads)
    LOGGER.debug(f"Memory available: {max_memory / 1E9:.2f} GB")
    LOGGER.debug(f"Using memory: {memory_to_use / 1E9:.2f} GB")
    LOGGER.debug(f"Number of threads: {n_of_threads}")
    LOGGER.debug(f"Chunk size per thread: {chunk_size_per_thread}")
    return chunk_size_per_thread


# Noting to parallelize in this function. It will raise an error if setting parallel to True since
# the calling function "chunked_dot" is already being parallelized.
@njit(parallel=False)
def _to_sparse(matrix, top_k):
    """Get the values and column indices of the biggest K elements per row.

    The function will return the data and indices according with the CSR matrix convention. The
    indptr values are calculated outside the parallelization of this function.
    """
    # This line creates a new array with the same shape as "matrix" effectively doubling the
    # memory consumption.
    top_k_j = np.argpartition(matrix, -top_k)
    if top_k > 0:
        top_k_j = top_k_j[:, -top_k:]
    else:
        top_k_j = top_k_j[:, :-top_k]
    values = np.take_along_axis(matrix, top_k_j, axis=1).flatten()
    indices = top_k_j.flatten()
    return values, indices


@njit(parallel=False)
def chunked_dot(matrix_left, matrix_right, top_k, chunk_size):
    """Parallelize the matrix multiplication by converting x into chunks."""
    n_rows = len(matrix_left)
    abs_top_k = abs(top_k)
    n_non_zero = n_rows * abs_top_k
    all_values, all_indices, all_indptr = (
        np.zeros(n_non_zero, dtype="float64"),
        np.empty(n_non_zero, dtype="int64"),
        np.empty(n_rows + 1, dtype="int64"),
    )
    all_indptr[0] = 0
    # Round up since the last's iteration chunk <= chunk_size
    for i in range(0, math.ceil(n_rows / chunk_size)):  # pylint: disable=not-an-iterable
        start_row_i, end_row_i = i * chunk_size, (i + 1) * chunk_size
        chunk_m = np.dot(matrix_left[start_row_i:end_row_i], matrix_right)
        values, indices = _to_sparse(chunk_m, top_k)
        all_values[start_row_i * abs_top_k : end_row_i * abs_top_k] = values
        all_indices[start_row_i * abs_top_k : end_row_i * abs_top_k] = indices
    # standard CSR form representation
    all_indptr = np.arange(0, abs_top_k * (1 + n_rows), abs_top_k)
    return all_values, all_indices, all_indptr


def warm_up_chunked_dot():
    """Make a dummy run of the "chunked_dot" function to compile it."""
    matrix = np.random.randn(10000, 100)
    chunked_dot(matrix, matrix.T, 10, 5000)


def cosine_similarity_top_k(
    embedings, top_k=None, max_memory_to_use=None, force_memory=False, float_type="float64"
):
    """Calculate cosine similarity and only keep the K most similar items for each item.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix containing non-zero values only for the K most
            similar items per item.

    This will:
        1. Normalize the rows in the "embeddings" matrix to have unit L2 norm.
        2. Calculate the optimal number of rows per piece when splitting the embeddings matrix.
        3. Parallelize the matrix multiplication using a separate piece in each thread.
        4. Per thread:
            a. Matrix multiplication between the piece and the embeddings matrix transposed.
            b. Extract the values and column indices of the most similar K items per row.
            c. Collect such values and column indices into outer scope arrays.
        5. Create a CSR matrix from all values and indices and return it.
    """

    embedings = embedings.astype(float_type)
    l2_norms = np.sqrt(np.einsum("ij,ij->i", embedings, embedings))
    embedings = embedings / l2_norms[:, np.newaxis]
    n_rows = embedings.shape[0]
    if top_k is None or top_k == n_rows:
        result = csr_matrix(np.dot(embedings, embedings.T))
    elif top_k > n_rows:
        raise ValueError("Requested more similar items than available items.")
    else:
        chunk_size_per_thread = get_chunk_size_per_thread(
            n_rows, top_k, max_memory_to_use, force_memory
        )
        values, indices, indptr = chunked_dot(embedings, embedings.T, top_k, chunk_size_per_thread)
        result = csr_matrix((values.astype(float_type), indices, indptr), shape=(n_rows, n_rows))
    return result
