from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi


def norm_per_axis(array: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Returns the norm of the vector or along some axis of an array.
    Default behaviour: if axis not specified, normalise a 1D vector or normalise 2D array row-wise. If axis specified,
    axis=0 normalises column-wise and axis=1 row-wise.

    Args:
        array: numpy array containing a vector or a set of vectors that should be normalised - per default assuming
               every row in an array is a vector
        axis: optionally specify along which axis the normalisation should occur

    Returns:
        an array of the same shape as the input array where each value is the norm of the corresponding vector/row/column
    """
    if axis is None:
        if len(array.shape) > 1:
            axis = 1
        else:
            axis = 0
    my_norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return np.repeat(my_norm, array.shape[axis], axis=axis)


def normalise_vectors(array: np.ndarray, axis: int = None, length=1) -> np.ndarray:
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
    Same as dist_on_sphere, but accepts and returns arrays and should only be used for already unitary vectors.

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


def cart2sph(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Transform an individual 3D point from cartesian to spherical coordinates.

    Code obtained from Leon Wehrhan.
    """
    XsqPlusYsq = x ** 2 + y ** 2
    r = np.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))  # theta
    az = np.arctan2(y, x)  # phi
    return r, elev, az


def cart2sphA(pts: NDArray) -> NDArray:
    """
    Transform an array of shape (N, 3) in cartesian coordinates to an array of the same shape in spherical coordinates.
    Can be used to create a hammer projection plot. In this case, disregard column 0 of the output and plot columns
    1 and 2.

    Code obtained from Leon Wehrhan.
    """
    return np.array([cart2sph(x, y, z) for x, y, z in pts])


def standardise_quaternion_set(quaternions: NDArray, standard=np.array([1, 0, 0, 0])) -> NDArray:
    """
    Since quaternions double-cover rotations, standardise all quaternions in this array so that their dot product with
    the "standard" quaternion is positive. Return the quaternion array where some elements ma have been flipped to -q.

    Idea: method 2 described here: https://math.stackexchange.com/questions/3888504/component-wise-averaging-of-similar-quaternions-while-handling-quaternion-doubl

    Args:
        quaternions: array (N, 4), each row a quaternion
        standard: a single quaternion determining which half-hypersphere to use
    """
    assert len(quaternions.shape) == 2 and quaternions.shape[1] == 4, "Wrong shape for quaternion array"
    assert standard.shape == (4, ), "Wrong shape for standard quaternion"
    # Take the dot product of q1 with all subsequent quaternions qi, for 2≤i≤N, and negate any of the subsequent
    # quaternions whose dot product with qi is negative.
    dot_product = standard.dot(quaternions.T)
    standard_quat = quaternions.copy()
    standard_quat[np.where(dot_product < 0)] = -quaternions[np.where(dot_product < 0)]

    assert standard_quat.shape == quaternions.shape
    return standard_quat


def randomise_quaternion_set_signs(quaternions: NDArray) -> NDArray:
    """
    Return the set of the same quaternions up to the sign of each row, which is normalised.
    """
    assert len(quaternions.shape) == 2 and quaternions.shape[1] == 4, "Wrong shape for quaternion array"
    random_set = np.random.choice([-1, 1], size=(len(quaternions), 1))
    re_signed = quaternions * random_set
    assert re_signed.shape == quaternions.shape
    return re_signed


def random_sphere_points(n: int = 1000) -> NDArray:
    """
    Create n points that are truly randomly distributed across the sphere. Eg. to test the uniformity of your grid.

    Args:
        n: number of points

    Returns:
        an array of grid points, shape (n, 3)
    """
    phi = np.random.uniform(0, 2*pi, (n, 1))
    costheta = np.random.uniform(-1, 1, (n, 1))
    theta = np.arccos(costheta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.concatenate((x, y, z), axis=1)


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
