from typing import Union

import numpy as np

from scipy import sparse

from chunkdot.chunkdot import chunkdot
from chunkdot.chunkdot_sparse import chunkdot_sparse
from chunkdot.utils import get_chunk_size_per_thread, normalize_embeddings


def cosine_similarity_top_k(
    embeddings: Union[np.ndarray, sparse.spmatrix],
    *,
    top_k: int,
    embeddings_right: Union[np.ndarray, sparse.spmatrix] = None,
    normalize: bool = True,
    max_memory: int = None,
    force_memory: bool = False,
    show_progress: bool = False,
):
    """Calculate cosine similarity and only keep the K most similar items for each item.

    Args:
        embeddings (np.array or scipy.sparse matrix): 2D object containing the items embeddings,
            of shape number of items x embedding dimension.
        top_k (int): The amount of similar items per item to return.
        embeddings_right (np.array or scipy.sparse matrix): 2D object containing the items
            embeddings used for cosine similarity calculations. If None, items in `embeddings` will
            be compared with themselves.
            Default None.
        normalize (bool): If to apply L2-norm to each row.
            Default True.
        max_memory (int): Maximum amount of memory to use in bytes. If None it will use the
            available memory to the system according to chunkdot.utils.get_memory_available.
            Default None.
        force_memory (bool): Use max_memory even if it is bigger than the memory
            available. This can be desired if the cosine similarity calculation is used many times
            within the same Python process, such that objects are garbage collected but memory is
            not marked as available to the OS. In this case is advised to set max_memory
            to chunkdot.utils.get_memory_available at the start of your Python process.
            Default False.
        show_progress (bool): Whether to show tqdm-like progress bar
            on chunked matrix multiplications.
            Default False.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix containing non-zero values only for the K most
            similar items per item.

    Raises:
        ValueError:
        TypeError:

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
    # return type consistent with sklearn.pairwise.cosine_similarity function
    return_type = "float32" if embeddings.dtype == np.float32 else "float64"
    embeddings = embeddings.astype(return_type)
    if normalize:
        embeddings = normalize_embeddings(embeddings, return_type)

    if embeddings_right is not None:
        embeddings_right = embeddings_right.astype(return_type)
        if normalize:
            embeddings_right = normalize_embeddings(embeddings_right, return_type)
    else:
        embeddings_right = embeddings  # copy by reference

    n_rows_left = embeddings.shape[0]
    n_rows_right = embeddings_right.shape[0]
    abs_top_k = abs(top_k)

    if abs_top_k >= n_rows_right:
        raise ValueError(
            f"The number of requested similar items (top_k={abs_top_k}) must be less than the "
            f"number of items available for comparison ({n_rows_right})"
        )

    chunk_size_per_thread = get_chunk_size_per_thread(
        n_rows_left, n_rows_right, abs_top_k, max_memory, force_memory
    )

    if sparse.issparse(embeddings) and sparse.issparse(embeddings_right):
        similarities = chunkdot_sparse(
            embeddings, embeddings_right.T, top_k, chunk_size_per_thread, return_type, show_progress
        )
    elif not sparse.issparse(embeddings) and not sparse.issparse(embeddings_right):
        similarities = chunkdot(
            embeddings, embeddings_right.T, top_k, chunk_size_per_thread, return_type, show_progress
        )
    else:
        raise TypeError(
            "Both embeddings matrices must be the same type, either dense or sparse. Matrix "
            f"`embeddings` is of type {type(embeddings)} and "
            f"`embeddings_right` is of type {type(embeddings_right)}"
        )

    return similarities
