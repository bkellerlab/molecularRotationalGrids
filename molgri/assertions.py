"""
All boolean/assertion helper functions.

This module implements helper functions that can be used by different test/other functions. The determining
characteristics of all functions in this module is that they operate with booleans (either return a boolean
answer or assert that a condition is true.

Example functions that belong to this module:
 - check if two quaternion sets represent the same rotation
 - check if all norms of vectors in the array are similar
 - check if given points form a square/a cube ...
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.constants import pi
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

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

def k_is_a_row(my_array: NDArray, k: NDArray) -> bool:
    """
    Check if the row k occurs anywhere in the array up to floating point precision
    Args:
        my_array ():
        k ():
        atol ():
        rtol ():

    Returns:

    """
    is_array_with_d_dim_r_rows_c_columns(k, r=my_array.shape[1])
    return np.any(np.all(np.isclose(k, my_array), axis=1))

def which_row_is_k(my_array: NDArray, k: NDArray) -> ArrayLike:
    """
    returns all indices of rows in my_array that are equal (within floating point errors) to my_array.
    Args:
        my_array:
        k:

    Returns:

    """
    if not np.any(my_array) or not np.any(k):
        return None
    return np.nonzero(np.all(np.isclose(k, my_array), axis=1))[0]

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


def norm_per_axis(array: NDArray, axis: int = None) -> NDArray:
    """
    Returns the norm of the vector or along some axis of an array.
    Default behaviour: if axis not specified, normalise a 1D vector or normalise 2D array row-wise. If axis specified,
    axis=0 normalises column-wise and axis=1 row-wise.

    Args:
        array: numpy array containing a vector or a set of vectors that should be normalised - per default assuming
               every row in an array is a vector
        axis: optionally specify along which axis the normalisation should occur

    Returns:
        an array of the same shape as the input array where each value is the norm of the corresponding
        vector/row/column
    """
    if axis is None:
        if len(array.shape) > 1:
            axis = 1
        else:
            axis = 0
    my_norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return np.repeat(my_norm, array.shape[axis], axis=axis)