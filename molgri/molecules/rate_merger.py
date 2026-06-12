"""
This is an intermediate step between building a rate/transition matrix and calculating eigenvectors where we want to
remove the states with extremely high energy to mage eigendecomposition possible (and fast)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, diags_array



def msm_determine_indices_never_visited_states(transition_matrix: csr_array) -> NDArray:
    """
    We are interested in the indices of states that were never visited during the simulation. These will have
    completely empty rows and columns in the transition matrix since there is never any transition to or from there
    cells.

    Note that the original array isn't affected by the transformation. To delete empty rows and columns,
    use the function delete_rows_columns()

    Args:
        transition_matrix (csr_array): a sparse matrix of shape (N_gridpoints, N_gridpoints)

    Returns:
        an array of indices, each index representing the state that was never visited during the simulation

    """
    # rows that are never visited (this is equal to the columns that are never visited)
    row_nnz = np.diff(transition_matrix.indptr)
    empty_mask = (row_nnz == 0)
    empty_indices = np.where(empty_mask)[0]
    return empty_indices

def sqra_determine_indices_never_visited_states(rate_matrix: csr_array, cutting_factor) -> NDArray:
    """
    For the rate matrix, we want to remove the elements based on the diagonal elements. The diagonal elements are the
    normalization, they are always negative and describe how much probability density is flowing out of the cell. We
    set a cut at -10^100 - if that much density (or more, causing overflow and -np.inf as entry) is flowing out,
    this state is simply unreachable and not contributing to the state of the system.

    This may not be perfect, but because we cannot perform eigendecomposition when the matrix if full of NaNs,
    it is very much necessary.

    Args:
        rate_matrix (csr_array): a sparse matrix of shape (N_gridpoints, N_gridpoints)

    Returns:
        an array of sorted indices, the rows & columns with these indices will be cut out since they represent states
        with too high energies.
    """

    if cutting_factor=="None":
        too_large_diagonal = []
    else:
        too_large_diagonal = np.where(rate_matrix.diagonal() < -float(cutting_factor))[0]
    too_large_diagonal.sort()
    print("too_large_diagonal ", len(too_large_diagonal))
    return np.array(too_large_diagonal)

    mask = np.isinf(rate_matrix.data)
    rows = np.unique(np.searchsorted(rate_matrix.indptr[1:], np.where(mask)[0], side='right'))

    combined = set(too_large_diagonal).union(set(rows))
    combined = list(combined)
    combined.sort()
    combined = np.array(combined)
    return combined

def delete_rows_columns(transition_matrix: csr_array, msm_or_sqra: str, cutting_factor=None) -> tuple:
    """
    This is a general function that deletes rows and columns with given indices from a matrix. The row and column
    with the same index are always removed together (since we are interested in symmetric matrices, they are the same).

    The cells that are deleted are the ones corresponding to very high-energy states.

    Args:
        transition_matrix (csr_array): a sparse matrix of shape (N_gridpoints, N_gridpoints)
        msm_or_sqra (str): define which matrix type you are working on, affects the selection of high-energy states
        and the normalization

    Returns:
        a tuple (smaller_transition_matrix, indices_to_keep) where the first one is a modified transition matrix (
        still a sparse matrix) and the second one the list of all indices that remain (sorted)
    """


    # first determine the high-energy indices
    if msm_or_sqra == "msm":
        indices_to_remove = msm_determine_indices_never_visited_states(transition_matrix)
    elif msm_or_sqra == "sqra":
        indices_to_remove = sqra_determine_indices_never_visited_states(transition_matrix, cutting_factor)
    else:
        raise ValueError(f"The parameter msm_or_sqra must be 'msm' or 'sqra', not: {msm_or_sqra}")



    #result = transition_matrix.copy()
    to_keep = list(set(range(transition_matrix.shape[1])) - set(indices_to_remove))
    to_keep.sort()

    result = remove_rows_cols(transition_matrix, indices_to_remove)

    # # as csr array delete relevant columns
    # result = result[:, to_keep]
    # # as csc array delete relevant rows
    # result = result.tocsc()
    # result = result[to_keep, :]
    # # back to csr array for consistency
    # result = result.tocsr()

    # If we are just deleting empty rows/columns, renormalization is not needed for MSM, but we do it anyway in case
    # we decide to delete other row/column pairs in the future
    if msm_or_sqra == "msm":
        result = msm_normalize(result)
    elif msm_or_sqra == "sqra":
        result = sqra_normalize(result)
    else:
        raise ValueError(f"The parameter msm_or_sqra must be 'msm' or 'sqra', not: {msm_or_sqra}")
    return result, to_keep

def msm_normalize(my_matrix: csr_array) -> csr_array:
    """
    This is the general way to normalize a transition (MSM) matrix so that the sum of the columns equals 1.

    Args:
        my_matrix (csr_array): a sparse, diagonal-symmetric matrix

    Returns:
        a sparse, diagonal-symmetric matrix with columns normalized to sum of one.
    """
    sums = my_matrix.sum(axis=1)
    # to avoid dividing by zero
    sums[sums == 0] = 1
    # now dividing with counts (actually multiplying with inverse)
    diagonal_values = np.reciprocal(sums)
    diagonal_matrix = diags_array(diagonal_values, format='csr')
    # Left multiply the CSR matrix with the diagonal matrix
    return diagonal_matrix.dot(my_matrix)


def sqra_normalize(my_matrix: csr_array | NDArray) -> csr_array | NDArray:
    """
    The rate matrix (SQRA) is normalized so that the sum of the rows is zero. This is achieved by filling the
    diagonal with the negative sums of the columns.

    Args:
        my_matrix ( csr_array | NDArray ): a 2D digonal-symmetric array

    Returns:
        my-matrix normalized so that the sum of columns is zero
    """
    my_matrix.setdiag(0)
    sums = my_matrix.sum(axis=1)
    # diagonal matrix of negative row-sums
    if isinstance(my_matrix, csr_array):
        sum_diag = diags_array(-sums, format="csr")
    else:
        sum_diag = np.diag(-sums)
    all_together = my_matrix + sum_diag
    return all_together

def expand_eigenvector_to_full_length(values: NDArray, indices: NDArray, full_length: int):
    """
    Because we canculate eigenvectors on a reduced rate or transition matrix (removing states with too high energies)
    but we want to have consistent numbering of states we expand the eigenvector again by setting all ignored states
    to the value of 0.

    Args:
        values (NDArray): an array of eigenvector elements of length (N_reduced_size,)
        indices (NDArray): an array of indices that were kept from the original array of length (N_reduced_size,)
        full_length (int): the full length of the grid, N_gridpoints

    Returns:
        an array of length N_gridpoints that is filled with values from the values array at positions of indices and
        otherwise filled with zeros.
    """
    result = np.zeros(full_length, dtype=values.dtype)
    result[indices] = values
    return result


def remove_rows_cols(A, remove_idx):
    remove_idx = np.asarray(remove_idx)
    n = A.shape[0]

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[remove_idx] = False

    keep_rows = np.nonzero(keep_mask)[0]

    # map old column indices → new ones
    new_index = -np.ones(n, dtype=int)
    new_index[keep_rows] = np.arange(len(keep_rows))

    data = []
    indices = []
    indptr = [0]

    for r in keep_rows:
        start, end = A.indptr[r], A.indptr[r + 1]
        cols = A.indices[start:end]
        vals = A.data[start:end]

        keep = keep_mask[cols]

        data.extend(vals[keep])
        indices.extend(new_index[cols[keep]])

        indptr.append(len(data))

    return csr_array((data, indices, indptr),
                     shape=(len(keep_rows), len(keep_rows)))
