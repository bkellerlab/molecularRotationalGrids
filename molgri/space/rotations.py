from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


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


# TODO: find usages
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
    #print(grid_x[0], grid_y[0], grid_z[0])
    b = rot_matrices[0]
    #print("b", rot_matrices[0])
    assert len(rot_matrices) == len(grid_x) == len(grid_y) == len(grid_z)
    # rot_matrices is an array in which each "row" is a 3x3 rotational matrix
    # create a rotational object set from a stack of rotational matrices
    rotations = Rotation.from_matrix(rot_matrices)
    a = rotations.as_matrix()[0]
    vec = np.random.random((3, 1))
    #print(a.dot(vec), b.dot(vec))
    #print("a", rotations.as_matrix()[0])
    return rotations


# TODO: find usages
def grid2quaternion(grid_x: NDArray, grid_y: NDArray, grid_z: NDArray) -> NDArray:
    """
    See grid2rotation; this function only reformats the output as a (N, 4) array of quaternions.
    """
    rot_objects = grid2rotation(grid_x, grid_y, grid_z)
    quaternions = rot_objects.as_quat()
    assert quaternions.shape[1] == 4, "quaternions must have 4 dimensions"
    return quaternions


# TODO: find usages
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
    elif s == 0 and c == 1:
        # if sin = 0, meaning that there is no rotation (or half circle)
        my_matrix = np.eye(3)
    else:
        my_matrix = -np.eye(3)
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


if __name__ == "__main__":
    from molgri.space.rotobj import build_grid_from_name
    ico_grid = build_grid_from_name("ico_18", "o", use_saved=False).get_grid()
    g2r = grid2rotation(ico_grid, ico_grid, ico_grid)
    z_vec = np.array([0, 0, 1])
    rodrigue = two_vectors2rot(z_vec, ico_grid[0])
    print(g2r.as_matrix()[0])
    print(rodrigue)
