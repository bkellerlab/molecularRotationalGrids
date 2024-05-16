"""
This is an intermediate step between building a rate matrix and calculating eigenvectors where we wat to merge some
states of the rate matrix that have very small differences in energy.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

# helper functions
def find_el_within_nested_list(my_list: list[list[Any]], my_el: Any) -> NDArray:
    """
    my_list is a a nested list like this: [[0, 7], [2], [1, 3], [4], [5, 6, 8]]. my_el is an element that
    occurs within my_list. Return the index/indices of the sublist in which my_el occurs or an empty list.

    Args:
        my_list: a list where every element is a sublist (can be of different lengths)
        my_el: an element that may or may not occur in the list

    Returns:
        a list of indices of sublists in which my_el occurs. Can be an empty list.
    """
    return np.where([my_el in sublist for sublist in my_list])[0]


def merge_sublists(input_list: list[list[Any]]) -> list[list[Any]]:
    """
    Given is an input_list of sub-lists. We want to join all sublists that share at least one element (transitive
    property, if [a, c] and [c, b] -> [a, b, c]. In the output list there should be no duplicates.

    Args:
        input_list: list of lists, e.g. [[0, 7], [2, 7, 3], [6, 8], [3, 1]]

    Return:
        list of lists in which groups that share at least one element are joined, sorted and unique, e.g.
        [[0, 1, 2, 3, 7], [6, 8]]
    """
    merged = []
    copy_input_list = deepcopy(input_list)  # avoiding modifying input

    while len(copy_input_list) > 0:
        # for this element we are looking at which others should be grouped with it
        first_element = copy_input_list[0][0]
        # find sublists that directly include this element
        indices_group = list(find_el_within_nested_list(copy_input_list, first_element))
        previous_len = len(indices_group) - 1
        # these are actual elements discovered so far
        entire_group = list(np.unique([copy_input_list[i] for i in indices_group]))
        # now by transitive property interested in groups of other members
        # repeat as long as we are adding new group members
        while len(indices_group) > previous_len:
            previous_len = len(indices_group)
            for group_el in entire_group:
                indices_group.extend(list(find_el_within_nested_list(copy_input_list, group_el)))
            indices_group = list(np.unique(indices_group))
            for i in indices_group:
                entire_group.extend(copy_input_list[i])
            entire_group = list(np.unique(entire_group))

        # delete merged-up elements
        for index_i in indices_group[::-1]:
            copy_input_list.pop(index_i)

        merged.append(entire_group)
    return merged


def sqra_normalize(my_matrix: NDArray):
    # diagonal matrix of negative column-sums
    sum_diag = np.diag(-my_matrix.sum(axis=1))
    return my_matrix + sum_diag


def merge_matrix_cells(my_matrix: NDArray | csr_array, all_to_join: list[list],
                       index_list: Optional[list] = None) -> tuple:
    """
    Merge some of the cells of my_matrix. Which separate cells should be joined into one is given by indices in the
    sublists of the all_to_join list. The merged cells will now be represented by the summed cell at the
    position of the smallest index.

    Args:
        - my_matrix: (sparse) quadratic, symmetric matrix in which we want to merge some states
        - all_to_join: a list index sub-lists eg. [[12, 7], [0, 15], [7, 9], [0, 16, 3]]
                       if there are multiple elements in a sublist, they will all be merged together. Same if
                       there are multiple sublists with the same element
        - index_list: None if no previous merge has been done. It there were previous merges, input the output[1]
                      of this function so the list of which merged indices correspond to new indices stays updated.

    Returns:
        a square, symmetric matrix smaller or equal to the input. If input is sqra-normalized so is the output


    Procedure for creating merge matrix P:
    - create diagonal matrix same shape as my_matrix
    - delete column j
    - in column i add 1 in row j

    Merge matrix P is applied to the starting matrix M as: P^T M P
    """

    # ALL ASSERTIONS AND PREPARATIONS HERE

    if index_list is None:
        internal_index_list = [[i] for i in range(my_matrix.shape[0])]
    else:
        # don't wanna change the input
        internal_index_list = deepcopy(index_list)

    assert len(internal_index_list) == my_matrix.shape[0], "Len of index list must be same as len of matrix"

    # my_matrix must be a 2D quadratic, symmetric matrix that is sqra-normalized
    assert len(my_matrix.shape) == 2 and my_matrix.shape[0] == my_matrix.shape[1], "input not 2D or not quadratic"
    # assert np.allclose(my_matrix, my_matrix.T), "input not symmetric"
    # assert np.allclose(my_matrix.sum(axis=1), 0), "input not sqra-normalized"

    # sub-elements of to_join must be at least 2 integers
    assert np.all([len(to_join) >= 2 for to_join in all_to_join]), "need at least to integers in to_join"
    # if elements occur in several sublists, merge these in one (transitive property)
    all_to_join = merge_sublists(all_to_join)

    # Note: since elements may have been dropped already, we don't use to_join integers to index my_matrix directly
    # but always search for integers within sublists of index_list
    reindexing_to_join = []
    for to_join in all_to_join:
        reindexing_sublist = []
        for my_i in to_join:
            reindexing_sublist.extend(find_el_within_nested_list(internal_index_list, my_i))
        # sort and remove duplicates that may occur after reindexing
        reindexing_to_join.append(list(np.unique(reindexing_sublist)))

    # CREATE AND APPLY A MERGE MATRIX

    merge_matrix = np.eye(my_matrix.shape[0])

    # we use the smallest index of each sublist as collective index
    collective_index = [to_join[0] for to_join in reindexing_to_join]

    # the rest of the indices will be added to the collective index
    merged_indices = [to_join[1:] for to_join in reindexing_to_join]
    flat_merged_indices = [a for to_join in reindexing_to_join for a in to_join[1:]]

    # in columns of collective indices add 1 in rows of corresponding merged indices
    for i, ci in enumerate(collective_index):
        for mi in merged_indices[i]:
            merge_matrix[mi][ci] = 1

    # delete columns of merged_indices (this must happen after you added ones!)
    merge_matrix = np.delete(merge_matrix, flat_merged_indices, axis=1)

    if isinstance(my_matrix, csr_array):
        merge_matrix = csr_array(merge_matrix)

    # now apply merge_matrix^T @ my_matrix @ merge_matrix
    result = merge_matrix.T @ my_matrix @ merge_matrix

    # FINAL CHECKS

    assert result.shape[0] == my_matrix.shape[0] - len(flat_merged_indices), "len(result) not len(input) - len(my_j)"
    assert len(result.shape) == 2 and result.shape[0] == result.shape[1], "result not 2D or not quadratic"
    assert np.allclose(result, result.T), "result not symmetric"
    assert np.allclose(result.sum(axis=1), 0), f"result not sqra-normalized: {result}"

    # delete the redundand indices in inverse order (and be super careful not to mess up inexing in the process)
    to_append = [[] for _ in merged_indices]
    flat_merged_indices.sort()
    for mi in flat_merged_indices[::-1]:
        which_subappend = find_el_within_nested_list(merged_indices, mi)[0]
        internal_index_list.pop(mi)
        to_append[which_subappend].append(mi)

    # expand the collective index
    for i, ci in enumerate(collective_index):
        where_append = find_el_within_nested_list(internal_index_list, ci)[0]
        internal_index_list[where_append].extend(to_append[i])

        # sort to keep everything neat
        internal_index_list[where_append].sort()

    return result, internal_index_list



if __name__ == "__main__":
    assert np.allclose(find_el_within_nested_list([[0, 2], [7], [13, 4]], 18), np.array([]))
    assert np.allclose(find_el_within_nested_list([[0, 2], [7], [13, 18, 4]], 18), np.array([2]))
    assert np.allclose(find_el_within_nested_list([[-7, 0, 2], [7], [13, 4], [-7]], -7), np.array([0, 3]))

    assert np.all(merge_sublists([[0, 7], [2, 7, 3], [6, 8], [3, 1]]) == [[0, 1, 2, 3, 7], [6, 8]])
    assert np.all(merge_sublists([[8, 6], [2, 7, 3], [6, 8], [3, 1]]) == [[6, 8], [1, 2, 3, 7]])

    # we have cells 0, 1, 2, 3, 4 and rates between all combinations
    A = np.array([[0, 3, 7, 8, 4],
                  [3, 0, 1, 2, 5],
                  [7, 1, 0, 8, 7],
                  [8, 2, 8, 0, 1],
                  [4, 5, 7, 1, 0]])

    A = sqra_normalize(A)

    merge_test1 = merge_matrix_cells(A, [[0, 0], [0, 1], [3, 0], [2, 4]])
    assert np.allclose(merge_test1[0], np.array([[-26.,  26.], [ 26., -26.]]))
    assert np.all(merge_test1[1] == [[0, 1, 3], [2, 4]])

