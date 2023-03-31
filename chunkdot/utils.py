import logging
import math
import warnings
import psutil
import numba
import numpy as np
from chunkdot.chunkdot import chunkdot


LOGGER = logging.getLogger(__name__)


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


def warm_up_chunked_dot():
    """Make a dummy run of the "chunkdot" function to compile it."""
    matrix = np.random.randn(10000, 100)
    chunkdot(matrix, matrix.T, 10, 5000)


def get_chunk_size_per_thread(n_items, top_k, max_memory=None, force_memory=False):
    """Calculate the maximum row size of a matrix for a given memory threshold.

    This calculation is very specific to the cosine_similarity_top_k algorithm. Given the total
    number of items, the amount of similar items requested, the number of parallel threads to
    execute and the memory threshold to consume, calculate the maximum number of rows to process
    on each thread.

    The memory consumed by the cosine_similarity_top_k algorithm is:

    Memory = (
        chunk_size x n_items x n_threads
        + chunk_size x n_items x n_threads
        + n_items x top_k
        + n_items x top_k
        + n_items
    ) x 8 bytes

    This function returns the solution for chunk_size in the above equation.

    Args:
        n_items (int): The total number of items used to perform the similarity pairwise operation.
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
    numerator = memory_to_use - 8 * n_items * (2 * top_k + 1)
    denominator = 16 * n_threads * n_items
    chunk_size = math.floor(numerator / denominator)
    LOGGER.debug(f"Memory available: {memory_available / 1E9:.2f} GB")
    LOGGER.debug(f"Maximum memory to use: {memory_to_use / 1E9:.2f} GB")
    LOGGER.debug(f"Number of threads: {n_threads}")
    LOGGER.debug(f"Chunk size per thread: {chunk_size}")
    return chunk_size
