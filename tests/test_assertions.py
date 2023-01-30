import numpy as np
from scipy.constants import pi
from scipy.spatial.transform import Rotation

from molgri.space.utils import normalise_vectors
from molgri.assertions import all_row_norms_similar, is_array_with_d_dim_r_rows_c_columns, all_row_norms_equal_k, \
    form_square_array, form_cube


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


def test_form_square():
    # non- aligned with axis in 2D
    square_2d = np.array([[0, 0], [2, 1], [3, -1], [1, -2]])
    np.random.shuffle(square_2d)
    assert form_square_array(square_2d)

    # a square in 3D
    square_3d = np.array([[0, 0, 0], [6, 0, 0], [6, -np.sqrt(24), np.sqrt(12)], [0, -np.sqrt(24), np.sqrt(12)]])
    np.random.shuffle(square_3d)
    assert form_square_array(square_3d)

    # aligned with axis in 4D
    square_4d = np.array([[0, -40, 0, 40], [0, 40, 0, 40], [0, -40, 0, -40], [0, 40, 0, -40]])
    np.random.shuffle(square_4d)
    assert form_square_array(square_4d)

    # give some examples of points that are not square
    non_square_2d = np.array([[1, 1], [3, 1], [3, 3], [1, -1]])
    np.random.shuffle(non_square_2d)
    assert not form_square_array(non_square_2d)

    # triangle
    triangle_2d = np.array([[0, 0], [0, 1], [0, 1], [-np.sqrt(2), 0]])
    np.random.shuffle(triangle_2d)
    assert not form_square_array(triangle_2d)

    # rectangle
    rectangle_2d = np.array([[-1, -2], [-1, 0], [3, 0], [3, -2]])
    np.random.shuffle(rectangle_2d)
    assert not form_square_array(rectangle_2d)

    # not co-planar in 3D
    non_planar_3d = np.array([[0, 0, 0], [2, 0, 0], [1, 1, np.sqrt(2)], [0, 2, 0]])
    np.random.shuffle(non_planar_3d)
    assert not form_square_array(non_planar_3d)


def test_form_cube():
    # a normal 3D cube
    cube3D = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [2, 2, 0], [2, 0, 2], [0, 2, 2], [2, 2, 2]])
    assert form_cube(cube3D)

    # same but an extra dimension kept constant
    cube4D = np.array([[5, 0, 0, 0], [5, 2, 0, 0], [5, 0, 2, 0], [5, 0, 0, 2],
                       [5, 2, 2, 0], [5, 2, 0, 2], [5, 0, 2, 2], [5, 2, 2, 2]])
    assert form_cube(cube4D)

    # 3D but not all distances the same
    cube3D_wrong = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 2], [1, 2, 0], [1, 0, 2], [0, 2, 2], [1, 2, 2]])
    assert not form_cube(cube3D_wrong)

    # 3D but tilted
    np.random.shuffle(cube3D)
    rot = Rotation.from_euler("XYZ", [pi/3, -17*pi/6, -0.23])
    tilted_cube3d = rot.apply(cube3D)
    assert form_cube(tilted_cube3d)

    # 4D but not a cube
    cube4D_wrong = np.array([[5, 0, 0, 0], [3, 2, 0, 0], [5, 0, 2, 0], [3, 0, 0, 2],
                       [5, 2, 2, 0], [3, 2, 0, 2], [5, 0, 2, 2], [3, 2, 2, 2]])
    assert not form_cube(cube4D_wrong)
