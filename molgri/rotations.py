import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


def quaternion2grid(array_quaternions: np.ndarray) -> np.ndarray:
    """
    Take an array where each row is a quaternion and convert each rotation into a point on a unit sphere.

    Args:
        array_quaternions: an array (N, 4) where each row is a quaternion needed for a rotation

    Returns:
        an array (N, 3) where each row is a coordinate on a 3D sphere
    """
    base_vector = np.array([0, 0, 1])
    points = np.zeros(array_quaternions[:, 1:].shape)
    for i, quat in enumerate(array_quaternions):
        my_rotation = Rotation.from_quat(quat)
        points[i] = my_rotation.apply(base_vector)
    return points


def grid2quaternion(grid_3D):
    """
    Take an array where each row is a point on a unit sphere and convert each rotation into a set of euler_123 angles.

    Args:
        grid_3D: an array (N, 3) where each row is a coordinate on a 3D sphere

    Returns:
        an array (N, 3) where each row is a set of the 3 euler angles needed for a rotation
    """

    assert np.allclose(np.linalg.norm(grid_3D, axis=1), 1), "Points on a grid must be unit vectors!"
    base_vector = np.array([0, 0, 1])
    points = np.zeros((grid_3D.shape[0], 4))
    for i, point in enumerate(grid_3D):
        my_matrix = two_vectors2rot(base_vector, point)
        my_rotation = Rotation.from_matrix(my_matrix)
        points[i] = my_rotation.as_quat()
    return points


def euler2grid(array_euler_angles: np.ndarray) -> np.ndarray:
    """
    Take an array where each row is a set of euler_123 angles and convert each rotation into a point on a unit sphere.

    Args:
        array_euler_angles: an array (N, 3) where each row is a set of the 3 euler angles needed for a rotation

    Returns:
        an array (N, 3) where each row is a coordinate on a 3D sphere
    """
    base_vector = np.array([0, 0, 1])
    points = np.zeros(array_euler_angles.shape)
    for i, euler in enumerate(array_euler_angles):
        my_rotation = Rotation.from_euler("ZYX", euler)
        points[i] = my_rotation.apply(base_vector)
    assert np.allclose(np.linalg.norm(points, axis=1), 1), "Points on a grid must be unit vectors!"
    return points


def grid2euler(grid_3D: np.ndarray) -> np.ndarray:
    """
    Take an array where each row is a point on a unit sphere and convert each rotation into a set of euler_123 angles.

    Args:
        grid_3D: an array (N, 3) where each row is a coordinate on a 3D sphere

    Returns:
        an array (N, 3) where each row is a set of the 3 euler angles needed for a rotation
    """
    assert np.allclose(np.linalg.norm(grid_3D, axis=1), 1), "Points on a grid must be unit vectors!"
    base_vector = np.array([0, 0, 1])
    points = np.zeros(grid_3D.shape)
    for i, point in enumerate(grid_3D):
        my_matrix = two_vectors2rot(base_vector, point)
        my_rotation = Rotation.from_matrix(my_matrix)
        points[i] = my_rotation.as_euler("ZYX")
    np.nan_to_num(points, copy=False)
    return points


# ########################## HELPER FUNCTIONS ################################

def skew(x: np.ndarray) -> np.ndarray:
    """
    Take a matrix and return its skew matrix.

    Args:
        x: a matrix

    Returns:
        skew matrix, see structure below
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def two_vectors2rot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Take vectors x and y and return the rotational matrix that transforms x into y.

    Args:
        x: an array of shape (3,), first vector
        y: an array of shape (3,),  second vector

    Returns:
        a 3x3 rotational matrix
    """
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    assert y.shape == x.shape == (3, 1)
    assert np.isclose(np.linalg.norm(x), 1) and np.isclose(np.linalg.norm(y), 1)
    v = np.cross(x.T, y.T)
    s = np.linalg.norm(v)
    c = np.dot(x.T, y)[0, 0]
    if s != 0:
        my_matrix = np.eye(3) + skew(v[0]) + skew(v[0]).dot(skew(v[0])) * (1 - c) / s ** 2
    else:
        # if sin = 0, meaning that there is no rotation (or half circle)
        my_matrix = np.eye(3)
    return my_matrix


class Rotation2D:

    def __init__(self, alpha: float or ArrayLike):
        """
        Initializes 2D rotation matrix with an angle.

        Args:
            alpha: angle in radians
        """
        rot_matrix = np.array([[np.cos(alpha), np.sin(alpha)],
                               [-np.sin(alpha), np.cos(alpha)]])
        self.rot_matrix = rot_matrix

    def apply(self, vector_set: ArrayLike, inverse: bool = False) -> ArrayLike:
        """
        Applies 2D rotational matrix to a set of vectors of shape (N, 2)

        Args:
            vector_set: array (each column a vector) that should be rotated
            inverse: True if the rotation should be inverted

        Returns:
            rotated vector of shape (N, 2)
        """
        if inverse:
            inverted_mat = self.rot_matrix.T
            result = vector_set.dot(inverted_mat)
        else:
            result = vector_set.dot(self.rot_matrix)
        result = result.squeeze()
        return result
