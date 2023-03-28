import math
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix, issparse
from chunkdot import numba_argpartition  # pylint: disable=unused-import


# Noting to parallelize in this function. It will raise an error if setting parallel to True since
# the calling function "_chunkdot" is already being parallelized.
@njit(parallel=False)
def _to_sparse(matrix, top_k, values, indices):
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


@njit(parallel=True)
def _chunkdot(matrix_left, matrix_right, top_k, chunk_size):
    """Parallelize matrix multiplication by converting the left matrix into chunks."""
    n_rows = len(matrix_left)
    abs_top_k = abs(top_k)
    n_non_zero = n_rows * abs_top_k
    all_values, all_indices = (
        np.zeros(n_non_zero, dtype="float64"),
        np.empty(n_non_zero, dtype="int64"),
    )
    # Round up since the in the last iteration chunk <= chunk_size
    for i in prange(0, math.ceil(n_rows / chunk_size)):  # pylint: disable=not-an-iterable
        start_row_i, end_row_i = i * chunk_size, (i + 1) * chunk_size
        chunk_m = np.dot(matrix_left[start_row_i:end_row_i], matrix_right)
        _to_sparse(
            chunk_m,
            top_k,
            all_values[start_row_i * abs_top_k : end_row_i * abs_top_k],
            all_indices[start_row_i * abs_top_k : end_row_i * abs_top_k],
        )
    # standard CSR form representation
    all_indptr = np.arange(0, abs_top_k * (1 + n_rows), abs_top_k)
    return all_values, all_indices, all_indptr


def chunkdot(matrix_left, matrix_right, top_k, chunk_size, return_type="float64"):
    """Parallelize matrix multiplication by converting the left matrix into chunks.

    Args:
        matrix_left (np.array): The left matrix in the matrix multiplication operation.
        matrix_right (np.array): The right matrix in the matrix multiplication operation.
        top_k (int): Keep only the biggest K values per row in the resulting matrix.
        chunk_size (int): The number of rows in the matrix_left to use per parallelized
            calculation.
        return_type (str): The return type of the matrix elements.

    Returns:
        scipy.sparse.csr_matrix: The result of the matrix multiplication as a CSR sparse matrix.
    """
    if issparse(matrix_left) or issparse(matrix_right):
        raise TypeError("ChunkDot does not yet support SciPy sparse matrices as input.")

    n_rows, n_cols = matrix_left.shape[0], matrix_right.shape[1]
    values, indices, indptr = _chunkdot(matrix_left, matrix_right, top_k, chunk_size)
    return csr_matrix((values.astype(return_type), indices, indptr), shape=(n_rows, n_cols))
