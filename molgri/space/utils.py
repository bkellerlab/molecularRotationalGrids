"""
Useful functions for rotations and vectors.

Functions that belong in this module perform simple conversions, normalisations or assertions that are useful at various
points in the molgri.space subpackage.
"""
from typing import Tuple

import numpy as np
from molgri.assertions import all_row_norms_equal_k, is_array_with_d_dim_r_rows_c_columns, norm_per_axis
from numpy.typing import NDArray, ArrayLike
from scipy.constants import pi
from scipy.spatial.transform import Rotation


def normalise_vectors(array: NDArray, axis: int = None, length: float = 1) -> NDArray:
    """
    Returns the unit vector of the vector or along some axis of an array.
    Default behaviour: if axis not specified, normalise a 1D vector or normalise 2D array row-wise. If axis specified,
    axis=0 normalises column-wise and axis=1 row-wise.

    Args:
        array: numpy array containing a vector or a set of vectors that should be normalised - per default assuming
               every row in an array is a vector
        axis: optionally specify along which axis the normalisation should occur
        length: desired new length for all vectors in the array

    Returns:
        an array of the same shape as the input array where vectors are normalised, now all have length 'length'
    """
    assert length > 0, "Length of a vector cannot be negative"
    my_norm = norm_per_axis(array=array, axis=axis)
    return length * np.divide(array, my_norm)


def angle_between_vectors(central_vec: np.ndarray, side_vector: np.ndarray) -> np.array:
    """
    Having two vectors or two arrays in which each row is a vector, calculate all angles between vectors.
    For arrays, returns an array giving results like those:

    ------------------------------------------------------------------------------------
    | angle(central_vec[0], side_vec[0])  | angle(central_vec[0], side_vec[1]) | ..... |
    | angle(central_vec[1], side_vec[0])  | angle(central_vec[1], side_vec[1]  | ..... |
    | ..................................  | .................................  | ..... |
    ------------------------------------------------------------------------------------

    Angle between vectors equals the distance between two points measured on a surface of an unit sphere!

    Args:
        central_vec: first vector or array of vectors
        side_vector: second vector or array of vectors

    Returns:

    """
    assert central_vec.shape[-1] == side_vector.shape[-1], f"Last components of shapes of both vectors are not equal:" \
                                                     f"{central_vec.shape[-1]}!={side_vector.shape[-1]}"
    v1_u = normalise_vectors(central_vec)
    v2_u = normalise_vectors(side_vector)
    angle_vectors = np.arccos(np.clip(np.dot(v1_u, v2_u.T), -1.0, 1.0))
    return angle_vectors


def dist_on_sphere(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """

    Args:
        vector1: vector shape (n1, d) or (d,)
        vector2: vector shape (n2, d) or (d,)

    Returns:
        an array the shape (n1, n2) containing distances between both sets of points on sphere
    """
    norm1 = norm_per_axis(vector1)
    norm2 = norm_per_axis(vector2)
    assert np.allclose(norm1, norm2), "Both vectors/arrays of vectors don't have the same norms!"
    angle = angle_between_vectors(vector1, vector2)
    return angle * norm1


def unique_quaternion_set(quaternions: NDArray) -> NDArray:
    """
    Select only the "upper half" of hyperspherical points (quaternions that may be repeating). How selection is done:
    select a list of all quaternions that have non-negative first coordinate.
    Among the nodes with first coordinate equal zero, select only the ones with non-negative second coordinate etc.

    Args:
        quaternions: array (N, 4), each row a quaternion

    Returns:
        quaternions: array (M <= N, 4), each row a quaternion different from all other ones
    """
    # test input
    all_row_norms_equal_k(quaternions, 1)
    is_array_with_d_dim_r_rows_c_columns(quaternions, d=2, c=4)

    non_repeating_quaternions = []
    for i in range(4):
        for projected_point in quaternions:
            if np.allclose(projected_point[:i], 0) and projected_point[i] > 0:
                # the point is selected
                non_repeating_quaternions.append(projected_point)
    return np.array(non_repeating_quaternions)


def find_inverse_quaternion(q: NDArray) -> NDArray:
    assert q.shape == (4,)
    new_q = []
    for i in range(1, 5):
        if np.allclose(q[:i], 0):
            new_q.append(0)
        else:
            new_q.append(-q[i-1])
            new_q.extend(q[i:])
            break
    return np.array(new_q)

def random_sphere_points(n: int = 1000) -> NDArray:
    """
    Create n points that are truly randomly distributed across the sphere.

    Args:
        n: number of points

    Returns:
        an array of grid points, shape (n, 3)
    """
    z_vec = np.array([0, 0, 1])
    quats = random_quaternions(n)
    rot = Rotation.from_quat(quats)
    return rot.apply(z_vec)


def random_quaternions(n: int = 1000) -> NDArray:
    """
    Create n random quaternions

    Args:
        n: number of points

    Returns:
        an array of grid points, shape (n, 4)
    """
    result = np.zeros((n, 4))
    random_num = np.random.random((n, 3))
    result[:, 0] = np.sqrt(1 - random_num[:, 0]) * np.sin(2 * pi * random_num[:, 1])
    result[:, 1] = np.sqrt(1 - random_num[:, 0]) * np.cos(2 * pi * random_num[:, 1])
    result[:, 2] = np.sqrt(random_num[:, 0]) * np.sin(2 * pi * random_num[:, 2])
    result[:, 3] = np.sqrt(random_num[:, 0]) * np.cos(2 * pi * random_num[:, 2])
    assert result.shape[1] == 4
    return result


def distance_between_quaternions(q1: NDArray, q2: NDArray) -> ArrayLike:
    """
    Calculate the distance between two unit quaternions or the pairwise distances between two arrays of unit
    quaternions. Quaternion distance is like hypersphere distance, but also considers double coverage.
    Args:
        q1 (): array either of shape (4,) or (N, 4), every row has unit length
        q2 (): array either of shape (4,) or (N, 4), every row has unit length

    Returns:
        Float or an array of shape (N,) containing distances between unit quaternions.
    """
    if q1.shape == (4,) and q2.shape == (4,):
        theta = angle_between_vectors(q1, q2)
    elif q1.shape[1] == 4 and q2.shape[1] == 4 and q1.shape[0]==q2.shape[0]:
        theta = np.diagonal(angle_between_vectors(q1, q2))
    else:
        raise ValueError("Shape of quaternions not okay")
    # if the distance would be more than half hypersphere, use the smaller distance
    return np.where(theta > pi / 2, pi-theta, theta)