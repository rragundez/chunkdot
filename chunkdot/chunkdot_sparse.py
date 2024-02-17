import logging
import math

import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
from scipy.sparse import csr_matrix
from scipy.sparse import rand as srand

from chunkdot.utils import to_sparse

LOGGER = logging.getLogger(__name__)


def warm_up_chunkdot_sparse():
    """Make a dummy run of the "chunkdot" function to compile it."""
    matrix = srand(10000, 100, density=0.01, format="csr")
    chunkdot_sparse(matrix, matrix.T, 10, 5000)


# Noting to parallelize in this function. It will raise an error if setting parallel to True since
# the calling functions are already being parallelized.
@njit(parallel=False)
def sparse_dot(
    matrix_left_data,
    matrix_left_indices,
    matrix_left_indptr,
    left_n_rows,
    matrix_right_data,
    matrix_right_indices,
    matrix_right_indptr,
    right_n_cols,
):
    """Sparse matrix multiplication matrix_left x matrix_right.

    Both matrices must be in the CSR sparse format.
    Data, indices and indptr are the standard CSR representation where the column indices for row i
    are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in
    data[indptr[i]:indptr[i+1]].

    Args:
        matrix_left_data (numpy.array): Non-zero values of the left sparse matrix.
        matrix_left_indices (numpy.array): Column index of non-zero values of the left matrix.
        matrix_left_indptr (numpy.array): Array with the count of non-zero values per row.
        left_n_rows (int): The number of rows of the left matrix.
        matrix_right_data (numpy.array): Non-zero values of the right sparse matrix.
        matrix_right_indices (numpy.array): Column index of non-zero values of the right matrix.
        matrix_right_indptr (numpy.array): Array with the count of non-zero values per row.
        right_n_cols (int): The number of columns of the right matrix.

    Returns:
        numpy.array: 2D array with the result of the matrix multiplication.
    """
    values = np.zeros((left_n_rows, right_n_cols))
    for row_left in range(left_n_rows):
        for left_i in range(matrix_left_indptr[row_left], matrix_left_indptr[row_left + 1]):
            col_left = matrix_left_indices[left_i]
            value_left = matrix_left_data[left_i]
            for right_i in range(matrix_right_indptr[col_left], matrix_right_indptr[col_left + 1]):
                col_right = matrix_right_indices[right_i]
                value_right = matrix_right_data[right_i]
                values[row_left, col_right] += value_left * value_right
    return values


# Noting to parallelize in this function. It will raise an error if setting parallel to True since
# the calling functions are already being parallelized.
@njit(parallel=False)
def slice_csr_sparse(data, indices, indptr, start_row, end_row):
    """Get row-wise slice of a sparse matrix.

    Data, indices and indptr are the standard CSR representation where the column indices for row i
    are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in
    data[indptr[i]:indptr[i+1]].

    Args:
        data (numpy.array): Non-zero value of the sparse matrix.
        indices (numpy.array): Column positions of non-zero values.
        indptr (numpy.array): Array with the count of non-zero values per row.
        start_row (int): Index of the first row to include in the slice.
        end_row (int): Index of the last row to include in the slice.

    Returns:
        3 arrays containing the sparse CSR matrix information of the slice.
    """
    left_i = indptr[start_row]
    end_row = min(end_row, len(indptr) - 1)
    right_i = indptr[end_row]
    return (
        data[left_i:right_i],
        indices[left_i:right_i],
        indptr[start_row : end_row + 1] - indptr[start_row],
    )


@njit(parallel=True)
def _chunkdot_sparse_rowwise(  # pylint: disable=too-many-arguments
    matrix_left_data,
    matrix_left_indices,
    matrix_left_indptr,
    left_n_rows,
    matrix_right_data,
    matrix_right_indices,
    matrix_right_indptr,
    right_n_cols,
    top_k,
    chunk_size,
    progress_bar=None,
):
    """Parallelize the sparse matrix multiplication by converting the left matrix into chunks."""
    # pylint: disable=duplicate-code
    abs_top_k = abs(top_k)
    n_non_zero = left_n_rows * abs_top_k
    all_values, all_indices = (
        np.zeros(n_non_zero, dtype="float64"),
        np.empty(n_non_zero, dtype="int64"),
    )
    # Round up since the in the last iteration chunk <= chunk_size
    num_iterations = math.ceil(left_n_rows / chunk_size)
    for i in prange(0, num_iterations):  # pylint: disable=not-an-iterable
        start_row_i, end_row_i = i * chunk_size, (i + 1) * chunk_size
        data, indices, indptr = slice_csr_sparse(
            matrix_left_data, matrix_left_indices, matrix_left_indptr, start_row_i, end_row_i
        )
        chunk_m = sparse_dot(
            data,
            indices,
            indptr,
            len(indptr) - 1,
            matrix_right_data,
            matrix_right_indices,
            matrix_right_indptr,
            right_n_cols,
        )
        to_sparse(
            chunk_m,
            top_k,
            all_values[start_row_i * abs_top_k : end_row_i * abs_top_k],
            all_indices[start_row_i * abs_top_k : end_row_i * abs_top_k],
        )
        if progress_bar is not None:
            progress_bar.update(1)
    # standard CSR form representation
    all_indptr = np.arange(0, abs_top_k * (1 + left_n_rows), abs_top_k)
    return all_values, all_indices, all_indptr


def chunkdot_sparse(
    matrix_left, matrix_right, top_k, chunk_size, return_type="float64", show_progress=False
):
    """Parallelize sparse matrix multiplication by converting the left matrix into chunks.

    Args:
        matrix_left (scipy.sparse.csr_matrix): The left sparse matrix in the matrix multiplication
            operation.
        matrix_right (scipy.sparse.csr_matrix): The right sparse matrix in the matrix
            multiplication operation.
        top_k (int): Keep only the biggest K values per row in the resulting matrix.
        chunk_size (int): The number of rows in the matrix_left to use per parallelized
            calculation.
        return_type (str): The return type of the matrix elements.
        show_progress (bool): Whether to show tqdm-like progress bar
            for parallel chunking

    Returns:
        scipy.sparse.csr_matrix: The result of the matrix multiplication as a CSR sparse matrix.
    """
    left_n_rows = matrix_left.shape[0]
    right_n_cols = matrix_right.shape[1]
    if matrix_left.shape[1] != matrix_right.shape[0]:
        raise ValueError(
            "Incorrect matrix dimensions for matrix multiplication. Left matrix has shape="
            f"({matrix_left.shape}) and right matrix has shape={matrix_right.shape}"
        )
    # Algorithm is made for CSR sparse matrices only
    left_format = matrix_left.getformat()
    if left_format != "csr":
        LOGGER.debug(f"Converting left matrix format from {left_format.upper()} to CSR.")
        matrix_left = matrix_left.tocsr()

    right_format = matrix_right.getformat()
    if right_format != "csr":
        LOGGER.debug(f"Converting right matrix format from {right_format.upper()} to CSR.")
        matrix_right = matrix_right.tocsr()

    num_iterations = math.ceil(left_n_rows / chunk_size)
    progress_bar = ProgressBar(total=num_iterations) if show_progress else None
    values, indices, indptr = _chunkdot_sparse_rowwise(
        matrix_left.data,
        matrix_left.indices,
        matrix_left.indptr,
        left_n_rows,
        matrix_right.data,
        matrix_right.indices,
        matrix_right.indptr,
        right_n_cols,
        top_k,
        chunk_size,
        progress_bar=progress_bar,
    )

    if progress_bar:
        progress_bar.close()

    return csr_matrix(
        (values.astype(return_type), indices, indptr), shape=(left_n_rows, right_n_cols)
    )
