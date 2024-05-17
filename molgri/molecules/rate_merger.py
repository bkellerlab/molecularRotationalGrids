"""
This is an intermediate step between building a rate matrix and calculating eigenvectors where we wat to merge some
states of the rate matrix that have very small differences in energy.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional
from time import time
from datetime import timedelta

import numpy as np
import networkx
from networkx.algorithms.components.connected import connected_components

from molgri.molecules.transitions import SQRA, SimulationHistogram
from numpy.typing import NDArray
from scipy.sparse import csr_array, coo_array
from scipy.sparse.linalg import eigs
from scipy.constants import k as kB, N_A

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
    copy_input_list = deepcopy(input_list)  # avoiding modifying input

    def to_graph(l):
        G = networkx.Graph()
        for part in l:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        """
            treat `l` as a Graph and returns it's edges
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)

        for current in it:
            yield last, current
            last = current

    G = to_graph(copy_input_list)
    return [sorted(list(x)) for x in connected_components(G)]


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
    all_to_join = merge_sublists(all_to_join)
    # ALL ASSERTIONS AND PREPARATIONS HERE

    if index_list is None:
        internal_index_list = [[i] for i in range(my_matrix.shape[0])]
        reindexing_to_join = deepcopy(all_to_join)
    else:
        # don't wanna change the input
        internal_index_list = deepcopy(index_list)
        # Note: since elements may have been dropped already, we don't use to_join integers to index my_matrix directly
        # but always search for integers within sublists of index_list
        reindexing_to_join = []
        for to_join in all_to_join:
            reindexing_sublist = []
            for my_i in to_join:
                reindexing_sublist.extend(find_el_within_nested_list(internal_index_list, my_i))
            # sort and remove duplicates that may occur after reindexing
            reindexing_to_join.append(list(np.unique(reindexing_sublist)))

    assert len(internal_index_list) == my_matrix.shape[0], "Len of index list must be same as len of matrix"


    # CREATE AND APPLY A MERGE MATRIX

    # first build merge matrix as coo-array
    rows = list(range(my_matrix.shape[0]))
    columns = list(range(my_matrix.shape[0]))

    collective_index = [to_join[0] for to_join in reindexing_to_join]

    # the rest of the indices will be added to the collective index
    merged_indices = [to_join[1:] for to_join in reindexing_to_join]
    flat_merged_indices = [a for to_join in reindexing_to_join for a in to_join[1:]]

    for i, ci in enumerate(collective_index):
        for mi in merged_indices[i]:
            rows.append(mi)
            columns.append(ci)

    merge_matrix = coo_array(([1,]*len(rows), (rows, columns)), dtype=bool)
    merge_matrix = merge_matrix.tocsc()

    # as csr matrix delete relevant columns
    to_keep = list(set(range(my_matrix.shape[1])) - set(flat_merged_indices))
    merge_matrix = merge_matrix[:, to_keep]

    if not isinstance(my_matrix, csr_array):
        merge_matrix = merge_matrix.todense()

    # now apply merge_matrix^T @ my_matrix @ merge_matrix -> this is a fast operation
    result = my_matrix.dot(merge_matrix)
    result = merge_matrix.T.dot(result)

    # expand the collective index
    for i, ci in enumerate(collective_index):
        for mi in merged_indices[i]:
            internal_index_list[ci].extend(internal_index_list[mi])
        # sort to keep everything neat
        internal_index_list[ci].sort()

    # delete the redundand indices in inverse order (to not mess up inexing in the process)
    flat_merged_indices.sort()
    for mi in flat_merged_indices[::-1]:
        internal_index_list.pop(mi)

    return result, internal_index_list


# working with rate matrices
def determine_rate_cells_to_join(my_sh: SimulationHistogram, bottom_treshold=0.001, T=273):
    # calculate delta Vs
    h_ij = my_sh.full_grid.get_full_distances().tocoo()
    row_indices = h_ij.row
    column_indices = h_ij.col
    potentials = my_sh.get_magnitude_energy(energy_type="Potential")
    delta_potentials = np.abs((potentials[row_indices] - potentials[
        column_indices]) * 1000) / (kB * N_A * T)

    # get pairs of cells to join
    high_e_frames = np.where(delta_potentials < bottom_treshold)[0]
    high_cell_pairs = [[r, c] for r, c in zip(row_indices[high_e_frames], column_indices[high_e_frames])]

    return high_cell_pairs


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
    assert np.all(merge_test1[1] == [[0, 1, 3], [2, 4]]), merge_test1[1]

    merge_test2 = merge_matrix_cells(A, [[0, 3]])
    expected2 = np.array([[-25, 5, 15, 5],
                         [5, -11, 1, 5],
                         [15, 1, -23, 7],
                         [5, 5, 7, -17]])
    expected_index2 = [[0, 3], [1], [2], [4]]
    assert np.allclose(merge_test2[0], expected2)
    assert np.all(merge_test2[1] == expected_index2)

    # testing merging with sparse matrices
    A_sparse = csr_array(A)
    step1s, index1s = merge_matrix_cells(my_matrix=A_sparse, all_to_join=[[0, 3]], index_list=None)
    assert np.allclose(step1s.todense(), expected2)
    assert np.all(index1s == expected_index2)

    merge_test3 = merge_matrix_cells(A, [[1, 3, 0]])
    expected3 = np.array([[-26, 16, 10],
                          [16, -23, 7],
                          [10, 7, -17]])
    expected_index3 = [[0, 1, 3], [2], [4]]
    assert np.allclose(merge_test3[0], expected3)
    assert np.all(merge_test3[1] == expected_index3)

    # testing merges in multiple steps
    step1, index1 = merge_matrix_cells(my_matrix=A, all_to_join=[[0, 1]], index_list=None)
    expected4 = np.array([[-27,   8, 10,   9],
                           [ 8, -23,   8,   7],
                           [10,   8, -19,   1],
                          [ 9,   7,   1, -17]])
    expected_index4 = [[0, 1], [2], [3], [4]]
    assert np.allclose(step1, expected4)
    assert np.all(index1 == expected_index4)
    step2, index2 = merge_matrix_cells(my_matrix=step1, all_to_join=[[3, 4]], index_list=index1)
    expected5 = np.array( [[-27.,   8.,  19.],
                           [  8., -23.,  15.],
                           [ 19.,  15., -34.]])
    expected_index5 = [[0, 1], [2], [3, 4]]
    assert np.allclose(step2, expected5)
    assert np.all(index2 == expected_index5)

    # this should work whether we specify [1, 3], [3, 0], [1, 3, 4], [0, 1, 3, 4] and similar
    step3a, index3a = merge_matrix_cells(my_matrix=step2, all_to_join=[[1, 3]], index_list=index2)
    expected6 = np.array([[-23, 23],
                          [23, -23]])
    expected_index6 = [[0, 1, 3, 4], [2]]
    assert np.allclose(step3a, expected6)
    assert np.all(index3a == expected_index6)
    step3b, index3b = merge_matrix_cells(my_matrix=step2, all_to_join=[[3, 0]], index_list=index2)
    assert np.allclose(step3b, expected6)
    assert np.all(index3b == expected_index6)
    step3c, index3c = merge_matrix_cells(my_matrix=step2, all_to_join=[[3, 4, 1], [1, 0], [3, 1]], index_list=index2)
    assert np.allclose(step3c, expected6)
    assert np.all(index3c == expected_index6)
    step3d, index3d = merge_matrix_cells(my_matrix=step2, all_to_join=[[1, 0, 3, 4]], index_list=index2)
    assert np.allclose(step3d, expected6)
    assert np.all(index3d == expected_index6)

    # rate matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from molgri.space.utils import k_argmax_in_array


    #sqra_name = "H2O_H2O_0632"
    #sqra_use_saved = False
    sqra_name = "H2O_H2O_0585"
    sqra_use_saved = False
    water_sqra_sh = SimulationHistogram(sqra_name, "H2O", is_pt=True,
                                        second_molecule_selection="bynum 4:6", use_saved=sqra_use_saved)
    sqra = SQRA(water_sqra_sh, use_saved=sqra_use_saved)
    rate_matrix = sqra.get_transitions_matrix()

    # initial eigenval, eigenvec
    def analyse(ratemat, index_list=None):
        print(f"######################  NEW ANALYSIS: size {ratemat.shape}, type: {type(ratemat)}###############")
        if index_list is None:
            index_list = [[i] for i in range(ratemat.shape[0])]

        t1 = time()
        eigenval, eigenvec = eigs(ratemat.T, 6, tol=1e-5, maxiter=100000, which="LM", sigma=0)
        t2 = time()

        print(f"Eigendecomposition matrix {ratemat.shape}: ", end="")
        print(f"{timedelta(seconds=t2 - t1)} hours:minutes:seconds")


        eigenval = eigenval.real
        eigenvec = eigenvec.real

        print(np.round(eigenval, 4))
        for i in range(5):
            num_extremes = 40
            magnitudes = eigenvec.T[i]
            most_positive = k_argmax_in_array(magnitudes, num_extremes)
            original_index_positive = []
            for mp in most_positive:
                original_index_positive.extend(index_list[mp])
            original_index_positive = np.array(original_index_positive)
            most_negative = k_argmax_in_array(-magnitudes, num_extremes)
            original_index_negative = []
            for mn in most_negative:
                original_index_negative.extend(index_list[mn])
            original_index_negative = np.array(original_index_negative)
            print(f"In {i}.coverage eigenvector {num_extremes} most positive cells are "
                  f"{list(original_index_positive + 1)}")
            print(f"and most negative {list(original_index_negative + 1)}.")
        return eigenval

    #analyse(rate_matrix)

    bottom_tresholds = [0.001, 0.01, 0.05, 0.1, 0.5]
    eigenvalues = []
    total_times = []
    num_cells = []

    for bt in bottom_tresholds:
        t4 = time()
        # now joining
        rate_to_join = determine_rate_cells_to_join(water_sqra_sh, bottom_treshold=0.05)

        print(f"Determined {len(rate_to_join)} pairs of cells to join")

        new_rate_matrix, new_ind = merge_matrix_cells(my_matrix=rate_matrix, all_to_join=rate_to_join)
        t5 = time()

        print(f"Construction of merge matrix: ", end="")
        print(f"{timedelta(seconds=t5 - t4)} hours:minutes:seconds")

        eigenvalues.append(analyse(new_rate_matrix, index_list=new_ind))

        t6 = time()
        total_times.append(t6-t4)
        num_cells.append(new_rate_matrix.shape[0])

        print(f"Size of original rate matrix is {rate_matrix.shape}, of reduced rate matrix {new_rate_matrix.shape}.")


    # TODO: make a plot of eigenvalues vs (time/bottom_treshold/number of cells

    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    for bottom_treshold in bottom_tresholds:
        sns.lineplot(x=bottom_treshold, y=eigenvalues, ax=ax[0])
        sns.lineplot(x=bottom_treshold, y=total_times, ax=ax[1])
        sns.lineplot(x=bottom_treshold, y=num_cells, ax=ax[2])
    plt.savefig(f"output/plotting/test_rate_merger_{sqra_name}")












































