from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from molgri.space.utils import normalise_vectors


def rotation2grid(rotations: Rotation) -> Tuple[NDArray, ...]:
    """
    Convert a series of N rotational objects (represented as a scipy object Rotation) to three grids by applying
    the rotations to a unit vectors in x, y, z directions.  The grids can be saved and converted to a rotational
    object later if needed or used in grid form to get positional grids in spherical coordinates.

    Args:
        rotations: a series of N rotational objects (represented as a scipy object Rotation)

    Returns:
        a tuple of three numpy arrays, each of shape (N, 3)
    """

    basis = np.eye(3)

    result = []
    for basis_vector in basis:
        rotated_bv = rotation2grid4vector(rotations, basis_vector)
        # special case when only one rotation is applied - need to expand dimensions
        if rotated_bv.shape == (3,):
            rotated_bv = rotated_bv[np.newaxis, :]
        assert rotated_bv.shape[1] == 3
        result.append(rotated_bv)
    return tuple(result)


def rotation2grid4vector(rotations: Rotation, vector: NDArray) -> NDArray:
    return rotations.apply(vector)


def quaternion2grid(quaternions: NDArray) -> Tuple[NDArray, ...]:
    """
    See rotation2grid function. This is only a helper function that parsers quaternions as inputs.
    """
    assert quaternions.shape[1] == 4, "Quaternions must have 4 dimensions"
    rotations = Rotation.from_quat(quaternions)
    return rotation2grid(rotations)


def euler2grid(euler_angles: NDArray) -> Tuple[NDArray, ...]:
    """
    See rotation2grid function. This is only a helper function that parsers euler angles as inputs.
    """
    assert euler_angles.shape[1] == 3, "Euler angles must have the shape (N, 3)"
    rotations = Rotation.from_euler("ZYX", euler_angles)
    return rotation2grid(rotations)


def grid2rotation(grid_x: NDArray, grid_y: NDArray, grid_z: NDArray) -> Rotation:
    """
    Re-create a rotational object by using the (saved) grid_x, grid_y and grid_z projections. We are looking for an
    array of rotational matrices R that achieve
        R[i] (1, 0, 0)^T = grid_x[i]
        R[i] (0, 1, 0)^T = grid_y[i]
        R[i] (0, 0, 1)^T = grid_z[i]
    for each i in range(len(grids)). It is easy to show that

            grid_x[i][0]  grid_y[i][0]  grid_z[i][0]
    R[i] =  grid_x[i][1]  grid_y[i][1]  grid_z[i][1]
            grid_x[i][2]  grid_y[i][2]  grid_z[i][2]

    Args:
        grid_x: an array (N, 3) where each row is a coordinate on a 3D sphere created by projecting rotation on 1, 0, 0
        grid_y: an array (N, 3) where each row is a coordinate on a 3D sphere created by projecting rotation on 0, 1, 0
        grid_z: an array (N, 3) where each row is a coordinate on a 3D sphere created by projecting rotation on 0, 0, 1

    Returns:
        a list of length N where each element is a rotational object
    """
    assert np.allclose(np.linalg.norm(grid_x, axis=1), 1), "Points on a x grid must be unit vectors!"
    assert np.allclose(np.linalg.norm(grid_y, axis=1), 1), "Points on a y grid must be unit vectors!"
    assert np.allclose(np.linalg.norm(grid_z, axis=1), 1), "Points on a z grid must be unit vectors!"
    assert grid_x.shape[1] == 3, f"grid_x must be of shape (N, 3), not {grid_x.shape}"
    assert grid_y.shape[1] == 3, f"grid_y must be of shape (N, 3), not {grid_y.shape}"
    assert grid_z.shape[1] == 3, f"grid_z must be of shape (N, 3), not {grid_z.shape}"

    # re-create rotational matrices from the three directional grids
    rot_matrices = np.concatenate((grid_x, grid_y, grid_z), axis=1)
    rot_matrices = rot_matrices.reshape((-1, 3, 3))
    rot_matrices = rot_matrices.swapaxes(1, 2)
    assert len(rot_matrices) == len(grid_x) == len(grid_y) == len(grid_z)
    # rot_matrices is an array in which each "row" is a 3x3 rotational matrix
    # create a rotational object set from a stack of rotational matrices
    rotations = Rotation.from_matrix(rot_matrices)
    return rotations


def grid2quaternion(grid_x: NDArray, grid_y: NDArray, grid_z: NDArray) -> NDArray:
    """
    See grid2rotation; this function only reformats the output as a (N, 4) array of quaternions.
    """
    rot_objects = grid2rotation(grid_x, grid_y, grid_z)
    quaternions = rot_objects.as_quat()
    assert quaternions.shape[1] == 4, "quaternions must have 4 dimensions"
    return quaternions


def grid2euler(grid_x: NDArray, grid_y: NDArray, grid_z: NDArray) -> NDArray:
    """
    See grid2rotation; this function only reformats the output as a (N, 3) array of Euler angles
    """
    print("Warning! Euler angles are not uniquely reconstructed from saved grids (but rotational matrices are).")
    rot_objects = grid2rotation(grid_x, grid_y, grid_z)
    eulers = rot_objects.as_euler("ZYX")
    np.nan_to_num(eulers, copy=False)
    return eulers


# ########################## HELPER FUNCTIONS ################################

def skew(x: NDArray) -> NDArray:
    """
    Take a vector or an array of vectors and return its skew matrix/matrices.

    Args:
        x: a vector (3,) or (N, 3)

    Returns:
        skew matrix, see structure below
    """

    def skew_2D(m):
        return np.array([[0, -m[2], m[1]],
                         [m[2], 0, -m[0]],
                         [-m[1], m[0], 0]])

    assert x.shape == (3,) or (len(x.shape) == 2 and x.shape[1] == 3)

    if x.shape == (3,):
        return skew_2D(x)
    else:
        skew_matrix = np.zeros((len(x), 3, 3))
        for i, row in enumerate(x):
            skew_matrix[i] = skew_2D(row)
        return skew_matrix


def two_vectors2rot(x: NDArray, y: NDArray) -> NDArray:
    """
    Take vectors x and y (or arrays of vectors with the same number of elements and return the
    rotational matrix that transforms x into y.

    Args:
        x: an array of shape (3,), first vector, or an array of vectors of size (N, 3)
        y: an array of shape (3,),  second vector, or an array of vectors of size (N, 3)

    Returns:
        a 3x3 rotational matrix
    """
    # checking the inputs
    assert y.shape == x.shape, "Arrays must be of same size."
    if len(x.shape) == 1:
        assert x.shape == (3,)
    elif len(x.shape) == 2:
        assert x.shape[1] == 3
    else:
        raise AttributeError("Arrays must be 1D or 2D")

    # converting 1D vectors in (1, 3) shape
    if y.shape == x.shape == (3,):
        x = x[np.newaxis, :]
        y = y[np.newaxis, :]
        assert y.shape == x.shape == (1, 3)
    N = len(x)

    # if not normalised yet, normalise
    x = normalise_vectors(x, axis=1)
    y = normalise_vectors(y, axis=1)
    assert np.allclose(np.linalg.norm(x, axis=1), 1) and np.allclose(np.linalg.norm(y, axis=1), 1)

    v = np.cross(x, y)
    s = np.linalg.norm(v, axis=1)
    c = np.matmul(x, y.T)
    c = c.diagonal()
    np.seterr(all="ignore")
    factor = np.divide((1 - c), s ** 2)
    my_matrix = N_eye_matrices(N, d=3) + skew(v) + np.matmul(skew(v), skew(v)) * factor[:, np.newaxis, np.newaxis]
    np.seterr(all="warn")
    sinus_zero = s[:, np.newaxis, np.newaxis] == 0
    cosinus_one = c[:, np.newaxis, np.newaxis] == 1
    my_matrix = np.where(np.logical_and(sinus_zero, cosinus_one), N_eye_matrices(N, d=3), my_matrix)
    my_matrix = np.where(np.logical_and(sinus_zero, ~cosinus_one),
                         -N_eye_matrices(N, d=3), my_matrix)
    if len(x) == len(y) == 1:
        my_matrix = my_matrix[0]
        assert my_matrix.shape == (3, 3)
    else:
        assert my_matrix.shape == (len(x), 3, 3)
    return my_matrix


def N_eye_matrices(N, d=3):
    """
    Returns a (N, d, d) array in which each 'row' is an identity matrix.

    Args:
        N: number of rows
        d: dimension of the identity matrix

    Returns:

    """
    shape = (N, d, d)
    identity_d = np.zeros(shape)
    idx = np.arange(shape[1])
    identity_d[:, idx, idx] = 1
    return identity_d


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
