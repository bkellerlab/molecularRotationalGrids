import numpy as np

from molgri.space.utils import normalise_vectors
from molgri.assertions import all_row_norms_similar, is_array_with_d_dim_r_rows_c_columns, all_row_norms_equal_k


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