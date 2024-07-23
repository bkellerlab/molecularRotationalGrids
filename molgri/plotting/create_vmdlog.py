"""
Vmdlogs are provided to VMD after loading the files to display specific styles and structures.
"""
import numpy as np
from numpy.typing import NDArray

from molgri.space.utils import k_argmax_in_array


def case_insensitive_search_and_replace(file_read, file_write, all_search_word, all_replace_word):
    with open(file_read, 'r') as file:
        file_contents = file.read()
        for search_word, replace_word in zip(all_search_word, all_replace_word):
            file_contents = file_contents.replace(search_word, replace_word)

    with open(file_write, 'w') as file:
        file.write(file_contents)


def show_eigenvectors(path_vmd_script_template: str, path_output_script: str, eigenvector_array: NDArray,
                      num_eigenvec: int, num_extremes: int, index_list: list = None):
    """
    Create a vmdlog that shows most + and - parts of the first few eigenvectors

    Args:
        eigenvector_array (): 2D eigenvector array, shape (num_evec, len_matrix_side) <- if you have multiple due to
        tau array, select and input the correct one.

    Returns:

    """
    # find the most populated states
    all_lists_to_insert = []
    for i, eigenvec in enumerate(eigenvector_array.T[:num_eigenvec]):
        print(eigenvec.shape)
        magnitudes = eigenvec
        # zeroth eigenvector only interested in max absolute values
        if i == 0:
            # most populated 0th eigenvector
            most_populated = k_argmax_in_array(np.abs(eigenvec), num_extremes)
            original_index_populated = []
            if index_list is None:
                original_index_populated = most_populated
            else:
                for mp in most_populated:
                    original_index_populated.extend(index_list[mp])
            # sort so that more extreme values in the beginning
            my_argsort = np.argsort(magnitudes[original_index_populated])[::-1]
            all_lists_to_insert.append(original_index_populated[my_argsort])
        else:
            most_positive = k_argmax_in_array(eigenvec, num_extremes)
            if index_list is None:
                original_index_positive = np.array(most_positive)
            else:
                original_index_positive = []
                for mp in most_positive:
                    original_index_positive.extend(index_list[mp])
                original_index_positive = np.array(original_index_positive)
            my_argsort_pos = np.argsort(magnitudes[original_index_positive])[::-1]
            most_negative = k_argmax_in_array(-magnitudes, num_extremes)
            if index_list is None:
                original_index_negative = np.array(most_negative)
            else:
                original_index_negative = []
                for mn in most_negative:
                    original_index_negative.extend(index_list[mn])
            my_argsort_neg = np.argsort(magnitudes[original_index_negative])
            all_lists_to_insert.append(original_index_positive[my_argsort_pos])
            all_lists_to_insert.append(original_index_negative[my_argsort_neg])
    # adding 1 to all because VMD uses enumeration starting with 1
    all_lists_to_insert = [list(map(lambda x: x + 1, sublist)) for sublist in all_lists_to_insert]
    all_str_to_replace = [f"REPLACE{i}" for i in range(num_eigenvec * 2 - 1)]
    all_str_to_insert = [', '.join(map(str, list(el))) for el in all_lists_to_insert]
    case_insensitive_search_and_replace(path_vmd_script_template, path_output_script, all_str_to_replace, all_str_to_insert)


def show_eigenvectors_MSM(path_vmd_script_template: str, path_output_script: str, path_assignments: str,
                          eigenvector_array: NDArray, num_eigenvec: int, num_extremes: int):
    """
In msm, after finding most populated eigenvectors, you must look at frames that have been assigned to this state.
    """
    my_assignments = np.load(path_assignments)
    # find the most populated states
    all_lists_to_insert = []
    for i, eigenvec in enumerate(eigenvector_array.T[:num_eigenvec]):
        magnitudes = eigenvec
        # zeroth eigenvector only interested in max absolute values
        if i == 0:
            # most populated 0th eigenvector
            most_populated = k_argmax_in_array(np.abs(eigenvec), num_extremes)
            original_index_populated = []
            for mp in most_populated:
                original_index_populated.extend(np.random.choice(np.where(my_assignments==mp)[0], 10))
            # sort so that more extreme values in the beginning
            #my_argsort = np.argsort(magnitudes[original_index_populated])[::-1]
            all_lists_to_insert.append(original_index_populated)
        else:
            most_positive = k_argmax_in_array(eigenvec, num_extremes)
            original_index_positive = []
            for mp in most_positive:
                original_index_positive.extend(np.random.choice(np.where(my_assignments==mp)[0], 10))
            #my_argsort_pos = np.argsort(magnitudes[original_index_positive])[::-1]
            most_negative = k_argmax_in_array(-magnitudes, num_extremes)
            original_index_negative = []
            for mn in most_negative:
                original_index_negative.extend(np.random.choice(np.where(my_assignments==mn)[0], 10))
            #my_argsort_neg = np.argsort(magnitudes[original_index_negative])
            all_lists_to_insert.append(original_index_positive)
            all_lists_to_insert.append(original_index_negative)
    # adding 1 to all because VMD uses enumeration starting with 1
    all_lists_to_insert = [list(map(lambda x: x + 1, sublist)) for sublist in all_lists_to_insert]
    all_str_to_replace = [f"REPLACE{i}" for i in range(num_eigenvec * 2 - 1)]
    all_str_to_insert = [', '.join(map(str, list(el))) for el in all_lists_to_insert]
    case_insensitive_search_and_replace(path_vmd_script_template, path_output_script, all_str_to_replace, all_str_to_insert)

def show_assignments(path_vmd_script_template: str, path_output_script: str, assignment_array: NDArray):
    """
    Create a vmdlog that shows separately the frames that are assigned to particular values in the given assignment
    array.
    """
    # find the most populated states
    all_lists_to_insert = []
    unique_groups = np.unique(assignment_array)
    for el in unique_groups:
        all_lists_to_insert.append(np.where(assignment_array==el)[0])
    all_lists_to_insert = [list(map(lambda x: x + 1, sublist)) for sublist in all_lists_to_insert]
    all_str_to_replace = [f"REPLACE{i}" for i in range(len(unique_groups))]
    all_str_to_insert = [', '.join(map(str, list(el))) for el in all_lists_to_insert]
    case_insensitive_search_and_replace(path_vmd_script_template, path_output_script, all_str_to_replace, all_str_to_insert)