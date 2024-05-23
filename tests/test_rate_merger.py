import numpy as np
from scipy.sparse import csr_array

from molgri.molecules.rate_merger import find_el_within_nested_list, merge_sublists, merge_matrix_cells, \
    sqra_normalize, delete_rate_cells


def test_find_within_nested_list():
    assert np.allclose(find_el_within_nested_list([[0, 2], [7], [13, 4]], 18), np.array([]))
    assert np.allclose(find_el_within_nested_list([[0, 2], [7], [13, 18, 4]], 18), np.array([2]))
    assert np.allclose(find_el_within_nested_list([[-7, 0, 2], [7], [13, 4], [-7]], -7), np.array([0, 3]))

def test_merge_sublists():
    assert np.all(merge_sublists([[0, 7], [2, 7, 3], [6, 8], [3, 1]]) == [[0, 1, 2, 3, 7], [6, 8]])
    assert np.all(merge_sublists([[8, 6], [2, 7, 3], [6, 8], [3, 1]]) == [[6, 8], [1, 2, 3, 7]])



def test_merge_matrix_cells():
    merge_test1 = merge_matrix_cells(A, [[0, 0], [0, 1], [3, 0], [2, 4]])
    assert np.allclose(merge_test1[0], np.array([[-26., 26.], [26., -26.]]))
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
    expected4 = np.array([[-27, 8, 10, 9],
                          [8, -23, 8, 7],
                          [10, 8, -19, 1],
                          [9, 7, 1, -17]])
    expected_index4 = [[0, 1], [2], [3], [4]]
    assert np.allclose(step1, expected4)
    assert np.all(index1 == expected_index4)
    step2, index2 = merge_matrix_cells(my_matrix=step1, all_to_join=[[3, 4]], index_list=index1)
    expected5 = np.array([[-27., 8., 19.],
                          [8., -23., 15.],
                          [19., 15., -34.]])
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

def test_delete_cells():
    output1, reindex1 = delete_rate_cells(A, to_remove=[0])
    ex_output1 = np.array([[-8, 1, 2, 5],
                           [1, -16, 8, 7],
                           [2, 8, -11, 1],
                           [5, 7, 1, -13]])
    assert np.allclose(output1, ex_output1)
    assert np.all(reindex1 == [[1], [2], [3], [4]])

    # now try to delete cell 0 again -> nothing should happen
    output2, reindex2 = delete_rate_cells(output1, to_remove=[0], index_list=reindex1)
    assert np.allclose(output2, output1)
    assert np.all(reindex2 == reindex1)

    # delete multiple cells
    output3, reindex3 = delete_rate_cells(A, to_remove=[3, 0])
    ex_output3 = np.array([[-6, 1, 5],
                           [1, -8, 7],
                           [5, 7, -12]])
    assert np.allclose(output3, ex_output3)
    assert np.all(reindex3 == [[1], [2], [4]])

    # delete from a sparse_matrix


if __name__ == "__main__":
    # we have cells 0, 1, 2, 3, 4 and rates between all combinations
    A = np.array([[0, 3, 7, 8, 4],
                  [3, 0, 1, 2, 5],
                  [7, 1, 0, 8, 7],
                  [8, 2, 8, 0, 1],
                  [4, 5, 7, 1, 0]])

    A = sqra_normalize(A)
    A_sparse = csr_array(A)

    test_find_within_nested_list()
    test_merge_sublists()
    test_merge_matrix_cells()
    test_delete_cells()
