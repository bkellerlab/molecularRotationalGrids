from molgri.assertions import two_sets_of_quaternions_equal
from molgri.space.rotations import Rotation2D, quaternion2grid, \
    grid2quaternion, grid2rotation, rotation2grid, two_vectors2rot
from molgri.space.utils import normalise_vectors, random_quaternions
from scipy.spatial.transform import Rotation

from scipy.constants import pi
import numpy as np


def test_rotation_2D():
    # rotating counterclockwise
    rot = Rotation2D(pi/6)
    expected_matrix = np.array([[np.sqrt(3)/2, -1/2], [1/2, np.sqrt(3)/2]])
    assert np.allclose(rot.rot_matrix, expected_matrix)
    # rotating clockwise
    rot = Rotation2D(-pi/6)
    expected_matrix = np.array([[np.sqrt(3)/2, 1/2], [-1/2, np.sqrt(3)/2]])
    assert np.allclose(rot.rot_matrix, expected_matrix)
    # more than 180 degrees
    rot = Rotation2D(pi + pi / 6)
    expected_matrix = np.array([[-np.sqrt(3)/2, 1/2], [-1/2, -np.sqrt(3)/2]])
    assert np.allclose(rot.rot_matrix, expected_matrix)
    # apply to one vector
    vector1 = np.array([-1, 1])
    vector1 = normalise_vectors(vector1)
    rot = Rotation2D(np.deg2rad(45))
    rot_vector = rot.apply(vector1)
    expected_vector = np.array([-1, 0])
    assert np.allclose(rot_vector, expected_vector)
    # apply in inverse
    rot = Rotation2D(np.deg2rad(45))
    rot_vector = rot.apply(vector1, inverse=True)
    expected_vector = np.array([0, 1])
    assert np.allclose(rot_vector, expected_vector)
    # apply to one vector for > 360 degrees
    vector2 = np.array([0, -1])
    rot = Rotation2D(np.deg2rad(2*360+180+30))
    rot_vector = rot.apply(vector2)
    expected_vector = np.array([-1/2, np.sqrt(3)/2])
    assert np.allclose(rot_vector, expected_vector)
    # apply to several vectors
    vector_set = np.array([[-np.sqrt(2)/2, np.sqrt(2)/2], [0, -1], [np.sqrt(3)/2, 1/2]])
    assert vector_set.shape == (3, 2)
    rot = Rotation2D(np.deg2rad(45))
    rot_vector_set = rot.apply(vector_set)
    expected_vector_set = np.array([[-1, 0],
                                    [np.sqrt(2)/2, -np.sqrt(2)/2],
                                    [(-1 + np.sqrt(3))/(2 * np.sqrt(2)), (1 + np.sqrt(3))/(2 * np.sqrt(2))]])
    assert np.allclose(rot_vector_set, expected_vector_set)
    # several vectors - apply in inverse
    rot_vector_set_i = rot.apply(vector_set, inverse=True)
    expected_vector_set_i = np.array([[0, 1],
                                     [-np.sqrt(2)/2, -np.sqrt(2)/2],
                                     [(1 + np.sqrt(3))/(2*np.sqrt(2)), -(-1 + np.sqrt(3))/(2*np.sqrt(2))]])
    assert np.allclose(rot_vector_set_i, expected_vector_set_i)
    # several vectors > 360 deg in inverse
    rot = Rotation2D(np.deg2rad(3*360))
    rot_vector_set_i = rot.apply(vector_set, inverse=True)
    assert np.allclose(rot_vector_set_i, vector_set)


def test_rot2grid():
    # create some random rotations
    N = 500
    initial_rot = Rotation.random(N)
    grid_x, grid_y, grid_z = rotation2grid(initial_rot)
    # re-create rotational objects
    after_rot = grid2rotation(grid_x, grid_y, grid_z)
    assert np.allclose(initial_rot.as_matrix(), after_rot.as_matrix())


def test_quat2grid():
    # create some random quaternions
    initial_quaternions = random_quaternions(50)
    grid_x, grid_y, grid_z = quaternion2grid(initial_quaternions)
    # re-create quaternions
    after_quaternions = grid2quaternion(grid_x, grid_y, grid_z)
    assert two_sets_of_quaternions_equal(initial_quaternions, after_quaternions)



def test_grid2rot():
    N = 500
    initial_grid_x, initial_grid_y, initial_grid_z = create_random_grids(N)

    rotation = grid2rotation(initial_grid_x, initial_grid_y, initial_grid_z)

    new_grid_x, new_grid_y, new_grid_z = rotation2grid(rotation)

    assert np.allclose(initial_grid_x, new_grid_x)
    assert np.allclose(initial_grid_y, new_grid_y)
    assert np.allclose(initial_grid_z, new_grid_z)

    # one more cycle of conversion just to be sure

    rotation2 = grid2rotation(new_grid_x, new_grid_y, new_grid_z)

    new_grid_x2, new_grid_y2, new_grid_z2 = rotation2grid(rotation2)

    assert np.allclose(new_grid_x, new_grid_x2)
    assert np.allclose(new_grid_y, new_grid_y2)
    assert np.allclose(new_grid_z, new_grid_z2)

    assert np.allclose(rotation.as_matrix(), rotation2.as_matrix())


def test_grid2quat():
    N = 500
    initial_grid_x, initial_grid_y, initial_grid_z = create_random_grids(N)

    quaternions = grid2quaternion(initial_grid_x, initial_grid_y, initial_grid_z)

    new_grid_x, new_grid_y, new_grid_z = quaternion2grid(quaternions)

    assert np.allclose(initial_grid_x, new_grid_x)
    assert np.allclose(initial_grid_y, new_grid_y)
    assert np.allclose(initial_grid_z, new_grid_z)

    # one more cycle of conversion just to be sure

    quaternions2 = grid2quaternion(new_grid_x, new_grid_y, new_grid_z)

    new_grid_x2, new_grid_y2, new_grid_z2 = quaternion2grid(quaternions2)

    assert np.allclose(new_grid_x, new_grid_x2)
    assert np.allclose(new_grid_y, new_grid_y2)
    assert np.allclose(new_grid_z, new_grid_z2)

    assert two_sets_of_quaternions_equal(quaternions, quaternions2)



# ############################ HELPER FUNCTIONS ###################################


def create_random_grids(N: int = 500):
    initial_rot = Rotation.random(N)
    return rotation2grid(initial_rot)


def test_two_vectors2rot():
    # examples of single vectors
    x = np.array([1, 0, 0])
    z = np.array([0, 0, 1])
    rot_mat1 = two_vectors2rot(x, z)
    assert np.allclose(rot_mat1[:, 0], [0, 0, 1])

    for _ in range(15):
        x = (np.random.random((3,)) - 0.5) * 7
        y = (np.random.random((3,)) - 0.3) * 3
        rot_mat2 = two_vectors2rot(x, y)
        # check with dot product
        product = rot_mat2.dot(x.T)
        norm_product = normalise_vectors(product)
        norm_y = normalise_vectors(y)
        assert np.allclose(norm_product, norm_y)
        # check with rotation
        my_rotation = Rotation.from_matrix(rot_mat2)
        rotated_x = my_rotation.apply(x)
        norm_rotated_x = normalise_vectors(rotated_x)
        assert np.allclose(norm_rotated_x, norm_y)

    # examples of vector arrays
    for _ in range(1):
        x = (np.random.random((8, 3))) * 7
        y = (np.random.random((8, 3)) - 0.3) * 3
        norm_y = normalise_vectors(y)
        rot_mat2 = two_vectors2rot(x, y)
        # check with rotation
        my_rotation = Rotation.from_matrix(rot_mat2)
        rotated_x = my_rotation.apply(x)
        norm_rotated_x = normalise_vectors(rotated_x)
        assert np.allclose(norm_rotated_x, norm_y)