import importlib
import logging
import math
import warnings

import numba
import numpy as np
import psutil
from numba import njit
from scipy import sparse

LOGGER = logging.getLogger(__name__)


def normalize_embeddings(embeddings, return_type):
    """L2 normalization of each row.

    Args:
        embeddings (np.array or scipy.sparse matrix): 2D object containing the items embeddings,
            of shape number of items x embedding dimension.
        return_type (str): The return type of the matrix elements.

    Returns:
        np.array or scipy.sparse matrix: Embeddings matrix with each row L2 normalized.
    """
    if sparse.issparse(embeddings):
        norms = sparse.linalg.norm(embeddings, ord=2, axis=1)
        norms[norms == 0] = np.inf
        embeddings = sparse.diags(1 / norms) @ embeddings
    else:
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        embeddings = np.divide(
            embeddings,
            norms,
            out=np.zeros_like(embeddings, dtype=return_type),
            where=norms != 0,
        )
    return embeddings


# Noting to parallelize in this function. It will raise an error if setting parallel to True since
# the calling functions are already being parallelized.
@njit(parallel=False)
def to_sparse(matrix, top_k, values, indices):
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
    values[:] = np.take_along_axis(matrix, top_k_j, axis=1).flatten()
    indices[:] = top_k_j.flatten()


def get_memory_available():
    """Get the available memory to the OS.

    https://psutil.readthedocs.io/en/latest/#psutil.virtual_memory

    The memory that can be given instantly to processes without the system going into swap. This
    is calculated by summing different memory values depending on the platform and it is supposed
    to be used to monitor actual memory usage in a cross platform fashion.

    Note that if called inside a running Python process, this might not reflect the
    available memory on that process, as memory could have been used in the Python process
    and released for further use in the same Python process but not back to the OS even if it has
    been garbage collected.
    """
    return psutil.virtual_memory().available


def get_chunk_size_per_thread(
    n_items_left, n_items_right, top_k, max_memory=None, force_memory=False
):
    """Calculate the maximum row size of a matrix for a given memory threshold.

    This calculation is very specific to the cosine_similarity_top_k algorithm. Given the number
    of items in the left matrix, the number of items in the right matrix, the amount of similar
    items requested, the number of parallel threads to execute and the memory threshold to consume,
    calculate the maximum number of rows to process on each thread.

    The memory consumed by the cosine_similarity_top_k algorithm is:

    Memory = (
        n_items_left x embedding_dim  # left matrix size
        n_items_right x embedding_dim  # right matrix size if left matrix != right matrix
        chunk_size x n_items_right x n_threads  # similarity values per chunk
        + chunk_size x n_items_right x n_threads  # column indices of sorted similarities per row
        + n_items_left x top_k  # matrix with all similarity values
        + n_items_left x top_k  # matrix indptr according to CSR notation
        + n_items_left  # matrix row indices according to CSR notation
    ) x 8 bytes

    This function returns the solution for chunk_size in the above equation.

    Args:
        n_items_left (int): The number of items to be compared.
        n_items_right (int): The number of items to be compared to.
        top_k (int): The amount of similar items per item to return.
        max_memory (int): Maximum amount of memory to use in bytes.
        force_memory (bool): Use max_memory even if it is bigger than the memory
            available. This can be desired if the cosine similarity calculation is used many times
            within the same Python process, such that objects are garbage collected but memory is
            not marked as available to the OS. In this case is advised to set max_memory
            to chunkdot.utils.get_memory_available at the start of your Python process.

    Returns:
        int: The maximum number of rows to calculate per thread.

    Raises:
        ValueError: If max_memory is bigger than the available memory and force_memory is False.

    Warns:
        If max_memory is bigger than the available memory and force_memory is True.

    """
    memory_available = get_memory_available()
    memory_to_use = memory_available
    if max_memory:
        if max_memory > memory_available:
            message = (
                f"Requested memory to use {max_memory / 1E9:.2f}GB is bigger than "
                f"the system's available memory {memory_available / 1E9:.2f}GB."
            )
            if force_memory:
                warnings.warn(message)
                memory_to_use = max_memory
            else:
                raise ValueError(message)
        else:
            memory_to_use = max_memory

    n_threads = numba.get_num_threads()
    numerator = memory_to_use - 8 * (2 * top_k + 1) * n_items_left
    denominator = 16 * n_threads * n_items_right
    chunk_size = math.floor(numerator / denominator)
    LOGGER.debug(f"Memory available: {memory_available / 1E9:.2f} GB")
    LOGGER.debug(f"Maximum memory to use: {memory_to_use / 1E9:.2f} GB")
    LOGGER.debug(f"Number of threads: {n_threads}")
    LOGGER.debug(f"Chunk size per thread: {chunk_size}")
    min_memory_to_use = denominator + 8 * (2 * top_k + 1) * n_items_left
    if chunk_size <= 1:
        raise ValueError(
            "The available memory or `max_memory` argument is not big enough to process a single "
            "chunk. If you used the `max_memory` argument please increase it to a value equal or "
            f"bigger than {min_memory_to_use  / 1E6:.2f}MB."
        )
    return chunk_size


def is_package_installed(package_name):
    """Check if a package is installed in the current Python runtime.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False if it is not.
    """
    if importlib.util.find_spec(package_name) is None:
        exists = False
    else:
        exists = True
    return exists
