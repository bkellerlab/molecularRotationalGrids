from molgri.rotations import Rotation2D, grid2quaternion, quaternion2grid, grid2euler, euler2grid
from molgri.utils import normalise_vectors
from molgri.analysis import random_sphere_points

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


def test_conversion_quaternion_grid():
    # from points to quaternions and back
    n = 50
    points = random_sphere_points(n)
    qua = grid2quaternion(points)
    after_points = quaternion2grid(qua)
    assert np.allclose(points, after_points)
    # the other direction
    # algorithm for random quaternions
    N = 50
    result = np.zeros((N, 4))
    random_num = np.random.random((N, 3))
    result[:, 0] = np.sqrt(1 - random_num[:, 0]) * np.sin(2 * pi * random_num[:, 1])
    result[:, 1] = np.sqrt(1 - random_num[:, 0]) * np.cos(2 * pi * random_num[:, 1])
    result[:, 2] = np.sqrt(random_num[:, 0]) * np.sin(2 * pi * random_num[:, 2])
    result[:, 3] = np.sqrt(random_num[:, 0]) * np.cos(2 * pi * random_num[:, 2])
    points = quaternion2grid(result)
    after_result = grid2quaternion(points)
    points2 = quaternion2grid(after_result)
    after_result2 = grid2quaternion(points2)
    # quaternions before and after not always the same (double coverage), but the rotational action is the
    # same, as shown by rotation matrices or grid points
    # TODO: not sure if sufficiently tested, the nature of rotational object not really preserved
    # not true: np.allclose(result, after_result)
    assert np.allclose(points, points2)
    assert np.allclose(after_result2, after_result)


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