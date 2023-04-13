import math
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from chunkdot.utils import to_sparse


# Noting to parallelize in this function. It will raise an error if setting parallel to True since
# the calling functions are already being parallelized.
@njit(parallel=False)
def sparse_dot_rowwise(
    matrix_left_data,
    matrix_left_indices,
    matrix_left_indptr,
    matrix_right_data,
    matrix_right_indices,
    matrix_right_indptr,
):
    left_n_rows = len(matrix_left_indptr) - 1
    right_n_rows = len(matrix_right_indptr) - 1
    similarities = np.empty((left_n_rows, right_n_rows))
    for row_left in range(left_n_rows):  # loop over rows of left matrix
        for row_right in range(right_n_rows):  # loop over rows of right matrix
            value = 0
            # loop over indices that belong to each row for the left matrix
            for left_i in range(matrix_left_indptr[row_left], matrix_left_indptr[row_left + 1]):
                # loop over indices that belong to each row for the right matrix
                for right_i in range(
                    matrix_right_indptr[row_right], matrix_right_indptr[row_right + 1]
                ):
                    if matrix_left_indices[left_i] == matrix_right_indices[right_i]:
                        # both rows have a non-zero value at this column index
                        value += matrix_left_data[left_i] * matrix_right_data[right_i]
            similarities[row_left, row_right] = value
    return similarities


# Noting to parallelize in this function. It will raise an error if setting parallel to True since
# the calling functions are already being parallelized.
@njit(parallel=False)
def slice_csr_sparse(data, indices, indptr, start_row, end_row):
    left_i = indptr[start_row]
    end_row = min(end_row, len(indptr) - 1)
    right_i = indptr[end_row]
    return (
        data[left_i:right_i],
        indices[left_i:right_i],
        indptr[start_row : end_row + 1] - indptr[start_row],
    )


@njit(parallel=True)
def _chunkdot_sparse_rowwise(
    matrix_left_data,
    matrix_left_indices,
    matrix_left_indptr,
    matrix_right_data,
    matrix_right_indices,
    matrix_right_indptr,
    top_k,
    chunk_size,
):
    """Parallelize matrix multiplication by converting the left matrix into chunks."""
    n_rows = len(matrix_left_indptr) - 1
    abs_top_k = abs(top_k)
    n_non_zero = n_rows * abs_top_k
    all_values, all_indices = (
        np.zeros(n_non_zero, dtype="float64"),
        np.empty(n_non_zero, dtype="int64"),
    )
    # Round up since the in the last iteration chunk <= chunk_size
    for i in prange(0, math.ceil(n_rows / chunk_size)):  # pylint: disable=not-an-iterable
        start_row_i, end_row_i = i * chunk_size, (i + 1) * chunk_size
        data, indices, indptr = slice_csr_sparse(
            matrix_left_data, matrix_left_indices, matrix_left_indptr, start_row_i, end_row_i
        )
        chunk_m = sparse_dot_rowwise(
            data,
            indices,
            indptr,
            matrix_right_data,
            matrix_right_indices,
            matrix_right_indptr,
        )
        to_sparse(
            chunk_m,
            top_k,
            all_values[start_row_i * abs_top_k : end_row_i * abs_top_k],
            all_indices[start_row_i * abs_top_k : end_row_i * abs_top_k],
        )
    # standard CSR form representation
    all_indptr = np.arange(0, abs_top_k * (1 + n_rows), abs_top_k)
    return all_values, all_indices, all_indptr


def chunkdot_sparse(matrix_left, matrix_right, top_k, chunk_size, return_type="float64"):
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
    n_rows = matrix_left.shape[0]
    matrix_right = matrix_right.T
    if matrix_left.shape[1] != matrix_right.shape[1]:
        raise ValueError("Incorrect matrix dimensions")
    n_cols = matrix_right.shape[0]
    values, indices, indptr = _chunkdot_sparse_rowwise(
        matrix_left.data,
        matrix_left.indices,
        matrix_left.indptr,
        matrix_right.data,
        matrix_right.indices,
        matrix_right.indptr,
        top_k,
        chunk_size,
    )
    return csr_matrix((values.astype(return_type), indices, indptr), shape=(n_rows, n_cols))
