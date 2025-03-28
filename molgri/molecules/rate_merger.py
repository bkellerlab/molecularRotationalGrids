"""
This is an intermediate step between building a rate matrix and calculating eigenvectors where we wat to merge some
states of the rate matrix that have very small differences in energy.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import networkx
from networkx.algorithms.components.connected import connected_components
from numpy.typing import NDArray
from scipy.sparse import csr_array, coo_array, diags
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
    my_property, if [a, c] and [c, b] -> [a, b, c]. In the output list there should be no duplicates.

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


def sqra_normalize(my_matrix: NDArray | csr_array):
    sums = my_matrix.sum(axis=1)
    # diagonal matrix of negative column-sums
    if isinstance(my_matrix, csr_array):
        sum_diag = diags(-sums, format="csr")
    else:
        sum_diag = np.diag(-sums)
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
def determine_rate_cells_to_join(distances, potentials, bottom_treshold=0.001, T=273):
    # calculate delta Vs
    h_ij = distances.tocoo()
    row_indices = h_ij.row
    column_indices = h_ij.col
    delta_potentials = np.abs((potentials[row_indices] - potentials[
        column_indices]) * 1000) / (kB * N_A * T)

    # get pairs of cells to join
    high_e_frames = np.where(delta_potentials < bottom_treshold)[0]
    high_cell_pairs = [[r, c] for r, c in zip(row_indices[high_e_frames], column_indices[high_e_frames])]

    return high_cell_pairs


def determine_rate_cells_with_too_high_energy(my_energies, energy_limit: float = 10, T: float = 273,
                                              lower_bound_factor: float = 1.5, upper_bound_factor: float = 1.5):
    # WORK WITH ENERGY LIMIT NOW
    import pandas as pd
    import numpy as np

    thermal_energy_kJ_mol = kB*N_A*T / 1000
    outlier_indices = np.where(my_energies > energy_limit*thermal_energy_kJ_mol)[0]
    print(f"With limit {energy_limit} gonna cut {len(outlier_indices)} cells.")

    #values = my_energies*1000/(kB*N_A*T)
    #df = pd.DataFrame({'values': values}) # in J/mol

    # Compute Q1 (25th percentile) and Q3 (75th percentile)
    # Q1 = df['values'].quantile(0.25, interpolation='midpoint')
    # Q3 = df['values'].quantile(0.75, interpolation='midpoint')
    # IQR = Q3 - Q1  # Interquartile range
    #
    # # Define outlier bounds
    # lower_bound = Q1 - lower_bound_factor * IQR
    # upper_bound = Q3 + upper_bound_factor * IQR

    # Identify outliers (including NaNs and Infs)
    # df['outlier'] = (df['values'] < lower_bound) | (df['values'] > upper_bound) | df['values'].isna() | np.isinf(
    #     df['values'])
    #
    # outlier_indices = df[df['outlier']].index.tolist()

    #too_high = np.where(my_energies*1000/(kB*N_A*T) > energy_limit)[0]

    return outlier_indices


def delete_rate_cells(my_matrix: NDArray | csr_array, to_remove: list,
                      index_list: list = None):
    """
    Another method to make a rate matrix smaller. Determine cells with high energies and remove them as rows and
    columns of the rate matrix.

    Args:
        my_matrix ():
        my_energies: as output of GROMACS[kJ/mol]
        energy_limit (): unitless, as a multiple of k_BT

    Returns:

    """

    # ALL ASSERTIONS AND PREPARATIONS HERE

    if index_list is None:
        internal_index_list = [[i] for i in range(my_matrix.shape[0])]
        reindexing_to_join = deepcopy(to_remove)
    else:
        # don't wanna change the input
        internal_index_list = deepcopy(index_list)
        # Note: since elements may have been dropped already, we don't use to_join integers to index my_matrix directly
        # but always search for integers within sublists of index_list
        reindexing_to_join = []
        for el_to_remove in to_remove:
            reindexing_to_join.extend(find_el_within_nested_list(internal_index_list, el_to_remove))
            # sort and remove duplicates that may occur after reindexing
        reindexing_to_join = list(np.unique(reindexing_to_join))

    # DROP ROWS AND COLUMNS FROM RATE MATRIX
    # as csr matrix delete relevant columns

    result = my_matrix.copy()
    print(my_matrix.shape[1], len(reindexing_to_join))
    to_keep = list(set(range(my_matrix.shape[1])) - set(reindexing_to_join))
    result = result[:, to_keep]
    # as csc array delete relevant rows
    if isinstance(my_matrix, csr_array):
        result = result.tocsc()
    result = result[to_keep, :]

    if isinstance(my_matrix, csr_array):
        result = result.tocsr()

    # re-normalize
    result = sqra_normalize(result)

    # drop elemets from internal index list
    internal_index_list = [x for i, x in enumerate(internal_index_list) if i in to_keep]

    return result, internal_index_list
