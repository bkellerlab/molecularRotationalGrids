"""
This module implements helper functions that can be used by different test/other functions. The determining
characteristics of all functions in this module is that they operate with booleans (either return a boolean
answer or assert that a condition is true.

Example functions that belong to this module:
 - check if two quaternion sets represent the same rotation
 - check if all norms of vectors in the array are similar
 - check if given points form a square/a cube ...
"""

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from molgri.space.utils import norm_per_axis
from molgri.constants import UNIQUE_TOL

##########################################################################################################
#                                          SPACE TOOLS                                                   #
##########################################################################################################


def check_equality(arr1: NDArray, arr2: NDArray, atol: float = None, rtol: float = None) -> bool:
    """
    Use the numpy function np.allclose to compare two arrays and return True if they are all equal. This function
    is a wrapper where I can set my preferred absolute and relative tolerance
    """
    if atol is None:
        atol = 1e-8
    if rtol is None:
        rtol = 1e-5
    return np.allclose(arr1, arr2, atol=atol, rtol=rtol)


def all_row_norms_similar(my_array: NDArray, atol: float = None, rtol: float = None) -> NDArray:
    """
    Assert that in an 2D array each row has the same norm (up to the floating point tolerance).

    Returns:
        the array of norms in the same shape as my_array
    """
    is_array_with_d_dim_r_rows_c_columns(my_array, d=2)
    axis = 1
    all_norms = norm_per_axis(my_array, axis=axis)
    average_norm = np.average(all_norms)
    assert check_equality(all_norms, average_norm, atol=atol, rtol=rtol), "The norms of all rows are not equal"
    return all_norms


def all_row_norms_equal_k(my_array: NDArray, k: float, atol: float = None, rtol: float = None) -> NDArray:
    """
    Same as all_row_norms_similar, but also test that the norm equals k.
    """
    my_norms = all_row_norms_similar(my_array=my_array, atol=atol, rtol=rtol)
    assert check_equality(my_norms, np.array(k), atol=atol, rtol=rtol), "The norms are not equal to k"
    return my_norms


def is_array_with_d_dim_r_rows_c_columns(my_array: NDArray, d: int = None, r: int = None, c: int = None):
    """
    Assert that the object is an array. If you specify d, r, c, it will check if this number of dimensions, rows, and/or
    columns are present.
    """
    assert type(my_array) == np.ndarray, "The first argument is not an array"
    # only check if dimension if d specified
    if d is not None:
        assert len(my_array.shape) == d, f"The dimension of an array is not d: {len(my_array.shape)}=!={d}"
    if r is not None:
        assert my_array.shape[0] == r, f"The number of rows is not r: {my_array.shape[0]}=!={r}"
    if c is not None:
        assert my_array.shape[1] == c, f"The number of columns is not c: {my_array.shape[1]}=!={c}"


def all_rows_unique(my_array: NDArray, tol: int = UNIQUE_TOL):
    """
    Check if all rows of the array are unique up to tol number of decimal places.
    """
    my_unique = np.unique(my_array.round(tol), axis=0)
    difference = np.abs(len(my_array) - len(my_unique))
    assert len(my_array) == len(my_unique), f"{difference} elements of an array are not unique up to tolerance."


def form_square_array(my_array: NDArray, dec_places=7) -> bool:
    """
    From an array of exactly 4 points, determine if they form a square (can be in space and not on a 2D plane).

    Args:
        my_array: array of shape (4, d) where d is the number of dimensions.
        dec_places: to how many decimal places to round when determining uniqueness

    Returns:
        True if points form a square, else False
    """
    distances = cdist(my_array, my_array)
    return _form_square(my_array, distances, dec_places=dec_places)


def _form_square(points, distances, dec_places=7) -> bool:
    """
    Given a set of points and already known distances between all pairs, determine if points form a square.
    """
    is_array_with_d_dim_r_rows_c_columns(points, d=2, r=4)
    # in each row of distances, the values must be: 1x0, 2xa (side length), 1xnp.sqrt(2)*a (diagonal length)
    dists, counts = np.unique(np.round(distances[0], dec_places), return_counts=True)   # sorts smallest to largest
    # from the first row, determine a and d = np.sqrt(2)*a
    try:
        dist_a = dists[1]
        dist_d = np.sqrt(2) * dist_a
    # IndexErrors occur when not a correct number of unique values and indicate that the point array isn't a square
    except IndexError:
        return False
    # now check this for every row
    for i, row in enumerate(distances):
        dists, counts = np.unique(np.round(row, dec_places), return_counts=True)  # sorts smallest to largest
        try:
            once_dist_zero = np.isclose(dists[0], 0) and counts[0] == 1
            twice_dist_a = counts[1] == 2
            once_dist_d = counts[2] == 1 and np.isclose(dists[2], dist_d)
        except IndexError:
            return False
        # if any of the conditions do not apply, return False immediately
        if not (once_dist_d and twice_dist_a and once_dist_zero):
            return False
    return True


def form_cube(my_array: NDArray, test_angles=False) -> bool:
    """
    Similarly to form_square, check if the 8 points supplied as rows in an array form a cube. The points may have
    >= 3 dimensions.

    Args:
        my_array: 2-dimensional array with 8 rows and at least 3 columns. It should be checked if the points form a
                  3D cube
        test_angles: select True if you want a more precise test that also looks at angles

    Returns:
        True if points form a cube, else False
    """
    is_array_with_d_dim_r_rows_c_columns(my_array, d=2, r=8)
    assert my_array.shape[1] >= 3, f"Only {my_array.shape[1]} columns given. " \
                                   f"Points with <3 dimensions cannot possibly form a cube!"
    # idea source: https://math.stackexchange.com/questions/1629899/given-eight-vertices-how-to-verify-they-form-a-cube
    distances = distance_matrix(my_array, my_array)
    # side length is the shortest distance occurring in distance matrix (except for 0
    a = np.min(distances[np.nonzero(distances)])
    # 24 elements should be equal to a
    if not np.count_nonzero(np.isclose(distances, a)) == 24:
        return False
    # 24 elements are face diagonals
    if not np.count_nonzero(np.isclose(distances, a*np.sqrt(2))) == 24:
        return False
    # 8 elements should be volume diagonals
    if not np.count_nonzero(np.isclose(distances, a * np.sqrt(3))) == 8:
        return False
    # the side length is a and as a consequence, distances to other points should be: 1x0, 3xa, 3xsqrt(2)*a, 1xsqrt(3)*a
    for row in distances:
        if not np.allclose(sorted(row), sorted([0, a, a, a, a*np.sqrt(2), a*np.sqrt(2), a*np.sqrt(2), a*np.sqrt(3)])):
            return False
    # only selected angles possible
    if test_angles:
        tetrahedron_a1 = np.arccos(-1 / 3)
        tetrahedron_a2 = np.arccos(1 / 3)
        all_angle_choices = [0, np.pi/2, np.pi, np.pi/3, tetrahedron_a1, tetrahedron_a2]
        for vec1 in my_array:
            for vec2 in my_array:
                angle_points = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
                if not np.any(np.isclose(angle_points, all_angle_choices, atol=1e-7)):
                    return False
    # finally, if all tests right
    return True


def quaternion_in_array(quat: NDArray, quat_array: NDArray) -> bool:
    """
    Check if a quaternion q or its equivalent complement -q is present in the quaternion array quat_array.
    """
    quat1 = quat[np.newaxis, :]
    for quat2 in quat_array:
        if two_sets_of_quaternions_equal(quat1, quat2[np.newaxis, :]):
            return True
    return False


def two_sets_of_quaternions_equal(quat1: NDArray, quat2: NDArray) -> bool:
    """
    This test is necessary because for quaternions, q and -q represent the same rotation. You therefore cannot simply
    use np.allclose to check if two sets of rotations represented with quaternions are the same. This function checks
    if all rows of two arrays are the same up to a flipped sign.
    """
    assert quat1.shape == quat2.shape
    assert quat1.shape[1] == 4
    # quaternions are the same if they are equal up to a +- sign
    # I have checked this fact and it is mathematically correct
    for q1, q2 in zip(quat1, quat2):
        if not (np.allclose(q1, q2) or np.allclose(q1, -q2)):
            return False
    return True


def assert_two_sets_of_eulers_equal(euler1: NDArray, euler2: NDArray):
    # TODO: can you check more than that?
    # see: https://en.wikipedia.org/wiki/Euler_angles#Signs,_ranges_and_conventions
    assert euler1.shape == euler2.shape
    assert euler1.shape[1] == 3
    # first of all, check that the corresponding rot matrices are the same
    rot1 = Rotation.from_euler("ZYX", euler1)
    rot2 = Rotation.from_euler("ZYX", euler2)
    assert np.allclose(rot1.as_matrix(), rot2.as_matrix())

    # TODO: check for gimbal locks

    # further checks
    for e1, e2 in zip(euler1, euler2):
        # if within [-pi/2, pi/2], euler angles must match exactly
        for comp1, comp2 in zip(e1, e2):
            if -pi/2 <= comp1 <= pi/2 and -pi/2 <= comp2 <= pi/2:
                assert np.isclose(comp1, comp2)
        # first and last angle must match up to pi
        assert np.isclose(e1[0] % pi, e2[0] % pi)
        assert np.isclose(e1[-1] % pi, e2[-1] % pi)