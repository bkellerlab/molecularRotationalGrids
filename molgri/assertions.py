"""
This module implements helper functions that can be used by different test functions. Should all assert
if sth is true and do not need to return anything.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from molgri.space.utils import norm_per_axis
from molgri.constants import UNIQUE_TOL

##########################################################################################################
#                                          SPACE TOOLS                                                   #
##########################################################################################################


def check_equality(arr1: NDArray, arr2: NDArray, atol: float = None, rtol: float = None) -> bool:
    """
    Use the numpy function np.allclose to compare two arrays and return if they are all equal.
    """
    if atol is None:
        atol = 1e-8
    if rtol is None:
        rtol = 1e-5
    return np.allclose(arr1, arr2, atol=atol, rtol=rtol)


def all_row_norms_similar(my_array: NDArray, atol: float = None, rtol: float = None):
    """
    Assert that in an 2D array each row has the same norm are similar up to the tolerance.

    Returns:
        the array of norms in the same shape as my_array
    """
    is_array_with_d_dim_r_rows_c_columns(my_array, d=2)
    axis = 1
    all_norms = norm_per_axis(my_array, axis=axis)
    average_norm = np.average(all_norms)
    assert check_equality(all_norms, average_norm, atol=atol, rtol=rtol), "The norms of all rows are not equal"
    return all_norms


def all_row_norms_equal_k(my_array: NDArray, k: float, atol: float = None, rtol: float = None):
    """
    Same as all_row_norms_similar, but also test that the norm equalts k.
    """
    my_norms = all_row_norms_similar(my_array=my_array, atol=atol, rtol=rtol)
    assert check_equality(my_norms, np.array(k), atol=atol, rtol=rtol), "The norms are not equal to k"


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


def form_square(point_array: NDArray, dec_places=7) -> bool:
    """
    From an array of exactly 4 points, determine if they form a square.

    Args:
        point_array: array of shape (4, d) where d is the number of dimensions.
        dec_places: to how many decimal places to round when determining uniqueness

    Returns:
        True if points form a square, else False
    """
    is_array_with_d_dim_r_rows_c_columns(point_array, d=2, r=4)
    distances = cdist(point_array, point_array)
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

