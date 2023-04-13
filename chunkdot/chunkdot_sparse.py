import logging
import math
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix, rand as srand
from chunkdot.utils import to_sparse


LOGGER = logging.getLogger(__name__)


def warm_up_chunkdot_sparse():
    """Make a dummy run of the "chunkdot" function to compile it."""
    matrix = srand(10000, 100, density=0.01, format="csr")
    chunkdot_sparse(matrix, matrix.T, 10, 5000)


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
    """Sparse matrix multiplication matrix_left x matrix_right.T

    Data, indices and indptr are the standard CSR representation where the column indices for row i
    are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in
    data[indptr[i]:indptr[i+1]].

    Note that the matrix_right would need still to be transposed according to the mathematical
    expression of matrix multiplication. In this algorithm we do not transpose it as the iteration
    is over the rows of each matrix.

    Args:
        matrix_left_data (numpy.array): Non-zero value of the sparse matrix.
        matrix_left_indices (numpy.array): Column positions of non-zero values.
        matrix_left_indptr (numpy.array): Array with the count of non-zero values per row.
        matrix_right_data (numpy.array): Non-zero value of the sparse matrix.
        matrix_right_indices (numpy.array): Column positions of non-zero values.
        matrix_right_indptr (numpy.array): Array with the count of non-zero values per row.

    Returns:
        numpy.array: 2D array with the result of the matrix multiplication.
    """
    left_n_rows = len(matrix_left_indptr) - 1
    right_n_rows = len(matrix_right_indptr) - 1
    values = np.empty((left_n_rows, right_n_rows))
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
            values[row_left, row_right] = value
    return values


# Noting to parallelize in this function. It will raise an error if setting parallel to True since
# the calling functions are already being parallelized.
@njit(parallel=False)
def slice_csr_sparse(data, indices, indptr, start_row, end_row):
    """Get row-wise slice of sparse matrix.

    Data, indices and indptr are the standard CSR representation where the column indices for row i
    are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in
    data[indptr[i]:indptr[i+1]].

    Args:
        data (numpy.array): Non-zero value of the sparse matrix.
        indices (numpy.array): Column positions of non-zero values.
        indptr (numpy.array): Array with the count of non-zero values per row.

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
    """Parallelize the sparse matrix multiplication by converting the left matrix into chunks.

    Note that the matrix_right would still need to be transposed according to the mathematical
    expression of matrix multiplication. In this algorithm we do not transpose it as the iteration
    is over the rows of each matrix.
    """
    # pylint: disable=too-many-locals, duplicate-code
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

    Returns:
        scipy.sparse.csr_matrix: The result of the matrix multiplication as a CSR sparse matrix.
    """
    if matrix_left.shape[1] != matrix_right.shape[0]:
        raise ValueError(
            "Incorrect matrix dimensions for matrix multiplication. Left matrix has shape="
            f"({matrix_left.shape}) and right matrix has shape={matrix_right.shape}"
        )

    n_rows = matrix_left.shape[0]
    matrix_right = matrix_right.T

    # Algorithm is made for CSR sparse matrices only
    left_format = matrix_left.getformat()
    if left_format != "csr":
        LOGGER.debug(f"Converting left matrix format from {left_format.upper()} to CSR.")
        matrix_left = matrix_left.tocsr()

    right_format = matrix_left.getformat()
    if right_format != "csr":
        LOGGER.debug(f"Converting right matrix format from {right_format.upper()} to CSR.")
        matrix_right = matrix_right.tocsr()

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
