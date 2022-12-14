from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def quaternion2grid(array_quaternions: NDArray, base_vector: NDArray) -> NDArray:
    """
    Take an array where each row is a quaternion and convert each rotation into a point on a unit sphere.

    Args:
        array_quaternions: an array (N, 4) where each row is a quaternion needed for a rotation

    Returns:
        an array (N, 3) where each row is a coordinate on a 3D sphere
    """
    assert array_quaternions.shape[1] == 4, "quaternions must have 4 dimensions"
    points = np.zeros(array_quaternions[:, 1:].shape)
    for i, quat in enumerate(array_quaternions):
        my_rotation = Rotation.from_quat(quat)
        points[i] = my_rotation.apply(base_vector)
    assert points.shape[1] == 3, "points on a 3D sphere must have 3 dimensions"
    return points


def quaternion2grid_z(array_quaternions: NDArray) -> NDArray:
    """
    This function applies the quaternion to a unit z-vector to create a grid.
    """
    base_vector = np.array([0, 0, 1])
    return quaternion2grid(array_quaternions, base_vector)


def quaternion2grid_x(array_quaternions: NDArray) -> NDArray:
    """
    This function applies the quaternion to a unit x-vector to create a grid.
    """
    base_vector = np.array([1, 0, 0])
    return quaternion2grid(array_quaternions, base_vector)


def grid2quaternion(grid_3D: NDArray, base_vector: NDArray) -> NDArray:
    """
    Take an array where each row is a point on a unit sphere and convert each rotation into an quaternion.

    Args:
        grid_3D: an array (N, 3) where each row is a coordinate on a 3D sphere
        base_vector: which base vector has been used to generate this grid

    Returns:
        an array (N, 4) where each row is a quaternion needed for a rotation
    """

    assert np.allclose(np.linalg.norm(grid_3D, axis=1), 1), "Points on a grid must be unit vectors!"
    assert grid_3D.shape[1] == 3, "points on a 3D sphere must have 3 dimensions"
    points = np.zeros((grid_3D.shape[0], 4))
    for i, point in enumerate(grid_3D):
        my_matrix = two_vectors2rot(base_vector, point)
        my_rotation = Rotation.from_matrix(my_matrix)
        points[i] = my_rotation.as_quat()
    assert points.shape[1] == 4, "quaternions must have 4 dimensions"
    return points


def grid2quaternion_z(grid_3D: NDArray) -> NDArray:
    """
    Convert a grid to a quaternion set by measuring the angle between the grid point and z-vector.
    """
    base_vector = np.array([0, 0, 1])
    return grid2quaternion(grid_3D, base_vector)


def grid2rotation(grid_3D: NDArray) -> List[Rotation]:
    assert np.allclose(np.linalg.norm(grid_3D, axis=1), 1), "Points on a grid must be unit vectors!"
    assert grid_3D.shape[1] == 3, "points on a 3D sphere must have 3 dimensions"
    base_vector = np.array([0, 0, 1])
    rotations = []
    for i, point in enumerate(grid_3D):
        my_matrix = two_vectors2rot(base_vector, point)
        rotations.append(Rotation.from_matrix(my_matrix))
    return rotations


def euler2grid(array_euler_angles: NDArray) -> NDArray:
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


def grid2euler(grid_3D: NDArray) -> NDArray:
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

def skew(x: NDArray) -> NDArray:
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


def two_vectors2rot(x: NDArray, y: NDArray) -> NDArray:
    # TODO: this could be done on arrays
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

    def __init__(self, alpha: float):
        """
        Initializes 2D rotation matrix with an angle. Rotates in counterclockwise direction.

        Args:
            alpha: angle in radians
        """
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                               [np.sin(alpha), np.cos(alpha)]])
        self.rot_matrix = rot_matrix

    def apply(self, vector_set: NDArray, inverse: bool = False) -> NDArray:
        """
        Applies 2D rotational matrix to a set of vectors of shape (N, 2) or a single vector with shape (2, )

        Args:
            vector_set: array (each row a vector) that should be rotated
            inverse: True if the rotation should be inverted (eg in clockwise direction)

        Returns:
            rotated vector set of the same shape as the initial vector set
        """
        if inverse:
            inverted_mat = self.rot_matrix.T
            result = inverted_mat.dot(vector_set.T)
        else:
            result = self.rot_matrix.dot(vector_set.T)
        result = result.squeeze()
        return result.T
