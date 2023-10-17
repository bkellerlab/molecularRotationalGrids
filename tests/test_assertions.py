import numpy as np
from scipy.constants import pi
from scipy.spatial.transform import Rotation

from molgri.space.utils import normalise_vectors
from molgri.assertions import all_row_norms_similar, is_array_with_d_dim_r_rows_c_columns, all_row_norms_equal_k, \
    form_square_array, form_cube, quaternion_in_array, two_sets_of_quaternions_equal


def test_all_row_norms_similar():
    lengths = (0.3, 17, 5)
    Ns = (55, 27, 89)
    for l, N in zip(lengths, Ns):
        random_arr = np.random.random((N, 3))  # between 0 and 1
        random_arr = normalise_vectors(random_arr, length=l)
        all_row_norms_similar(random_arr)
        all_row_norms_equal_k(random_arr, l)


def test_is_array_with_d_dim_r_rows_c_columns():
    my_arr = np.random.random((5, 3))
    is_array_with_d_dim_r_rows_c_columns(my_arr, d=2, r=5, c=3)
    my_arr2 = np.random.random((7, 5, 3, 1))
    is_array_with_d_dim_r_rows_c_columns(my_arr2, d=4, r=7, c=5)
    is_array_with_d_dim_r_rows_c_columns(my_arr2, d=4)
    is_array_with_d_dim_r_rows_c_columns(my_arr2, r=7, c=5)
    my_arr3 = np.random.random((3,))
    is_array_with_d_dim_r_rows_c_columns(my_arr3, d=1, r=3)



def test_quat_in_array():
    q1 = np.array([5, 7, -3, -1])
    q_array = np.array([[12, -1, -0.3, -0.2], [5, 7, -3, -1], [-0.1, 0.1, 2, 3], [-12, 1, 0.3, 0.2]])
    # test q is in array
    assert quaternion_in_array(q1, q_array)
    # test -q is in array
    q2 = np.array([0.1, -0.1, -2, -3])
    assert quaternion_in_array(q2, q_array)
    # test multiple occurences
    q3 = np.array([-12, 1, 0.3, 0.2])
    assert quaternion_in_array(q3, q_array)
    # test not in array
    q4 = np.array([-12, -1, -0.3, -0.2])
    assert not quaternion_in_array(q4, q_array)


def test_quaternion_sets_equal():
    # true positive
    quat_set_1 = np.random.random((15, 4))
    quat_set_2 = np.copy(quat_set_1)
    quat_set_2[5] = -quat_set_1[5]
    quat_set_2[12] = -quat_set_1[12]
    assert two_sets_of_quaternions_equal(quat_set_1, quat_set_2)
    # true negative
    quat_set_3 = np.copy(quat_set_1)
    quat_set_3[8][0] = -quat_set_1[8][0]
    assert not two_sets_of_quaternions_equal(quat_set_1, quat_set_3)