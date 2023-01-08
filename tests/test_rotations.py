from molgri.rotations import Rotation2D, grid2quaternion_z, quaternion2grid_z, grid2euler, euler2grid, quaternion2grid, \
    grid2quaternion
from molgri.utils import normalise_vectors
from molgri.analysis import random_sphere_points, random_quaternions
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


def test_point2qua_and_back():
    # from points to quaternions and back
    n = 50
    points = random_sphere_points(n)
    qua = grid2quaternion_z(points)
    after_points = quaternion2grid_z(qua)
    assert np.allclose(points, after_points)


def test_qua2_point_and_back():
    # algorithm for random quaternions

    # creating some random quaternions
    N = 50
    result = random_quaternions(N)

    # using a vector in z-direction as a base for creating a grid
    points_z = quaternion2grid_z(result)
    after_result = grid2quaternion_z(points_z)
    points_z2 = quaternion2grid_z(after_result)
    after_result2 = grid2quaternion_z(points_z2)

    # using a different vector as a base for creating a grid
    mixed_base = normalise_vectors(np.array([2, 7, -3])) # use instead of the z-vector for grid creation
    points_mix = quaternion2grid(result, mixed_base)
    after_result_mix = grid2quaternion(points_mix, None, None)
    points_mix_2 = quaternion2grid(after_result_mix, mixed_base)
    after_result_mix_2 = grid2quaternion(points_mix_2, None, None)
    # quaternions before and after not always the same (double coverage), but the rotational action is the
    # same, as shown by rotation matrices or grid points
    # TODO: not sure if sufficiently tested, the nature of rotational object not really preserved
    print("qua", result[0], after_result[0])
    print("qua", result[0], after_result_mix[0])
    # not true: np.allclose(result, after_result)
    assert np.allclose(points_z, points_z2)
    assert np.allclose(points_mix, points_mix_2)
    assert np.allclose(after_result2, after_result)
    assert np.allclose(after_result_mix, after_result_mix_2)


def test_new_qua2_point_and_back():
    N = 10
    initial_quaternions = random_quaternions(N)
    initial_rotation = Rotation(quat=initial_quaternions)

    # apply to all coordinate axes
    x_vec = np.array([1, 0, 0])
    y_vec = np.array([0, 1, 0])
    z_vec = np.array([0, 0, 1])

    x_rot = initial_rotation.apply(x_vec)
    y_rot = initial_rotation.apply(y_vec)
    z_rot = initial_rotation.apply(z_vec)

    # re-create the rotation only by using rotated vectors
    rotated_basis = np.concatenate((x_rot, y_rot, z_rot), axis=1)
    rotated_basis = rotated_basis.reshape((-1, 3, 3))
    rotated_basis = rotated_basis.swapaxes(1, 2)

    R_0 = np.array([x_rot[0], y_rot[0], z_rot[0]]).T
    print(R_0)
    print("rot basis", rotated_basis[0])

    assert np.allclose(initial_rotation.as_matrix(), rotated_basis)


def test_conversion_euler_grid():
    # from points to quaternions and back
    n = 50
    points = random_sphere_points(n)
    qua = grid2euler(points)
    after_points = euler2grid(qua)
    assert np.allclose(points, after_points)
    # other direction
    N = 50
    euler_angles = 2 * pi * np.random.random((N, 3))
    grid = euler2grid(euler_angles)
    euler_angles2 = grid2euler(grid)
    grid2 = euler2grid(euler_angles2)
    euler_angles3 = grid2euler(grid2)
    # TODO: same as with quaternions, not sure if this OK
    # not true: assert np.allclose(euler_angles, euler_angles2)
    assert np.allclose(grid, grid2)
    assert np.allclose(euler_angles2, euler_angles3)


def test_two_vectors_to_rot():
    pass