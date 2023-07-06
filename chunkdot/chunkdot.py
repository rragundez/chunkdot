import math

import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
from scipy.sparse import csr_matrix

from chunkdot.utils import to_sparse


def warm_up_chunkdot():
    """Make a dummy run of the "chunkdot" function to compile it."""
    matrix = np.random.randn(10000, 100)
    chunkdot(matrix, matrix.T, 10, 5000)


@njit(parallel=True)
def _chunkdot(matrix_left, matrix_right, top_k, chunk_size, progress_bar=None):
    """Parallelize matrix multiplication by converting the left matrix into chunks."""
    n_rows = len(matrix_left)
    abs_top_k = abs(top_k)
    n_non_zero = n_rows * abs_top_k
    all_values, all_indices = (
        np.zeros(n_non_zero, dtype="float64"),
        np.empty(n_non_zero, dtype="int64"),
    )
    # Round up since the in the last iteration chunk <= chunk_size
    num_iterations = math.ceil(n_rows / chunk_size)

    for i in prange(0, num_iterations):  # pylint: disable=not-an-iterable
        start_row_i, end_row_i = i * chunk_size, (i + 1) * chunk_size
        chunk_m = np.dot(matrix_left[start_row_i:end_row_i], matrix_right)
        to_sparse(
            chunk_m,
            top_k,
            all_values[start_row_i * abs_top_k : end_row_i * abs_top_k],
            all_indices[start_row_i * abs_top_k : end_row_i * abs_top_k],
        )
        if progress_bar is not None:  # if None is dropped, raises numba.core.errors.TypingError
            progress_bar.update(1)
    # standard CSR form representation
    all_indptr = np.arange(0, abs_top_k * (1 + n_rows), abs_top_k)
    return all_values, all_indices, all_indptr


def chunkdot(
    matrix_left, matrix_right, top_k, chunk_size, return_type="float64", show_progress=False
):
    """Parallelize matrix multiplication by converting the left matrix into chunks.

    Args:
        matrix_left (np.array): The left matrix in the matrix multiplication operation.
        matrix_right (np.array): The right matrix in the matrix multiplication operation.
        top_k (int): Keep only the biggest K values per row in the resulting matrix.
        chunk_size (int): The number of rows in the matrix_left to use per parallelized
            calculation.
        return_type (str): The return type of the matrix elements.
        show_progress (bool): Whether to show tqdm-like progress bar
            for parallel chunking

    Returns:
        scipy.sparse.csr_matrix: The result of the matrix multiplication as a CSR sparse matrix.
    """
    n_rows, n_cols = matrix_left.shape[0], matrix_right.shape[1]

    num_iterations = math.ceil(len(matrix_left) / chunk_size)
    progress_bar = ProgressBar(total=num_iterations) if show_progress else None
    values, indices, indptr = _chunkdot(
        matrix_left, matrix_right, top_k, chunk_size, progress_bar=progress_bar
    )

    if progress_bar:
        progress_bar.close()

    return csr_matrix((values.astype(return_type), indices, indptr), shape=(n_rows, n_cols))
