import numpy as np
from scipy.constants import pi

from molgri.space.utils import normalise_vectors, angle_between_vectors, dist_on_sphere, norm_per_axis
from molgri.space.analysis import random_sphere_points, vector_within_alpha, count_points_within_alpha, \
    random_axes_count_points
from molgri.space.rotobj import build_grid


def test_unit_dist_on_sphere():
    # angle between two vectors must be equal to distance times the radius of sphere
    for i in range(1, 20):
        vec1 = i*normalise_vectors((np.random.random(size=(3,)) - 0.5) * 2)
        vec2 = i*normalise_vectors((np.random.random(size=(3,)) - 0.5) * 2)
        angle = i*angle_between_vectors(vec1, vec2)
        dist = dist_on_sphere(vec1, vec2)
        assert np.allclose(angle, dist)


def test_unit_vector():
    """Testing that the axis argument is used as expected."""
    # example of a 2D array
    test_array = np.array([[-3, -0.2, -7],
                           [6, 5, -1],
                           [2, 9, 9],
                           [-18, 0.6, -0.3]])
    # axis = 0 is column-wise
    normalised_0 = normalise_vectors(test_array, axis=0)
    assert normalised_0.shape == test_array.shape, "Shape shouldn't change when normalising vectors"
    assert np.allclose(normalised_0[:, 0], np.array([-0.155334, 0.310668, 0.103556, -0.932005]))
    assert np.allclose(normalised_0[:, 1], np.array([-0.0193892, 0.484729, 0.872513, 0.0581675]))
    assert np.allclose(normalised_0[:, 2], np.array([-0.611383, -0.0873404, 0.786064, -0.0262021]))
    # axis = 1 should be row-wise normalisation
    normalised_1 = normalise_vectors(test_array, axis=1)
    assert normalised_1.shape == test_array.shape, "Shape shouldn't change when normalising vectors"
    assert np.allclose(normalised_1[0], np.array([-0.393784, -0.0262522, -0.918828]))
    assert np.allclose(normalised_1[1], np.array([0.762001, 0.635001, -0.127]))
    assert np.allclose(normalised_1[2], np.array([0.15523, 0.698535, 0.698535]))
    assert np.allclose(normalised_1[3], np.array([-0.999306, 0.0333102, -0.0166551]))
    # same if axis not defined
    normalised_1 = normalise_vectors(test_array)
    assert normalised_1.shape == test_array.shape, "Shape shouldn't change when normalising vectors"
    assert np.allclose(normalised_1[0], np.array([-0.393784, -0.0262522, -0.918828]))
    # example of a simple 1D array
    test_simple_array = np.array([2, -7, -0.9])
    norm = normalise_vectors(test_simple_array)
    assert np.allclose(norm, np.array([0.272646, -0.95426, -0.122691]))
    # same if axis defined:
    norm = normalise_vectors(test_simple_array, axis=0)
    assert np.allclose(norm, np.array([0.272646, -0.95426, -0.122691]))


def test_angle_between():
    # in 2D
    vec1 = np.array([0, 1])
    vec2 = np.array([1, 0])
    assert np.isclose(angle_between_vectors(vec1, vec2), pi/2)
    # in 3D, non-unitary vectors
    vec3 = np.array([0, -1, 0])
    vec4 = np.array([0, 2, 0])
    assert np.isclose(angle_between_vectors(vec3, vec4), pi)
    subvec_50 = [2, 7, 3]
    subvec_51 = [-1, 2, 4]
    subvec_60  = [2, 1, 5]
    subvec_61 = [-8, -0.8, 9]
    vec5 = np.array([subvec_50, subvec_51])
    vec6 = np.array([subvec_60, subvec_61])
    norm_50 = np.linalg.norm(subvec_50)
    norm_51 = np.linalg.norm(subvec_51)
    norm_60 = np.linalg.norm(subvec_60)
    norm_61 = np.linalg.norm(subvec_61)
    manual_angle_00 = np.arccos(np.dot(subvec_50, subvec_60)/(norm_60*norm_50))
    manual_angle_01 = np.arccos(np.dot(subvec_50, subvec_61)/(norm_61*norm_50))
    manual_angle_10 = np.arccos(np.dot(subvec_51, subvec_60) / (norm_60 * norm_51))
    manual_angle_11 = np.arccos(np.dot(subvec_51, subvec_61) / (norm_61 * norm_51))
    assert np.isclose(angle_between_vectors(vec5, vec6)[0, 0], manual_angle_00)
    assert np.isclose(angle_between_vectors(vec5, vec6)[0, 1], manual_angle_01)
    assert np.isclose(angle_between_vectors(vec5, vec6)[1, 0], manual_angle_10)
    assert np.isclose(angle_between_vectors(vec5, vec6)[1, 1], manual_angle_11)


def test_random_sphere_point():
    n = 25
    rsp = random_sphere_points(n)
    assert rsp.shape == (n, 3)
    assert np.allclose(norm_per_axis(rsp), 1)
    # TODO: test that it's a random distribution???


def test_vector_within_alpha():
    one_vector = np.array([0, 0, 1])
    another_vector = np.array([pi/6, pi/6, 1])
    assert vector_within_alpha(one_vector, another_vector, alpha=pi/4), "Vector within alpha doesn't work correctly" \
                                                                        "for 1D vectors"
    # for 2D arrays
    one_array = np.array([[0, 0, 1], [0.3, -0.3, 1.2]])
    another_array = np.array([[pi/6, pi/6, 1], [0.3, -0.3, -1.2]])
    assert np.all(vector_within_alpha(one_array, another_array, alpha=pi/4) == [[True, False], [True, False]])
    # also useful in mixed version - one constant vector and comparing angles from all others to it
    constant_z = np.array([0, 0, 1])
    various_others = np.array([[pi/6, pi/6, 1], [0.3, -0.3, -1.2], [-0.5, 0.5, 0.8]])
    within = vector_within_alpha(constant_z, various_others, alpha=pi/4)
    assert np.all(within == [True, False, True])
    assert within.shape == (various_others.shape[0],)


def test_count_within_alpha():
    # all points within 180 degrees
    num_points = 50
    grid_r = build_grid(num_points, "randomQ")
    points = grid_r.get_grid()
    assert count_points_within_alpha(points, np.array([0, 0, 1]), pi) == num_points
    assert count_points_within_alpha(points, np.array([pi/4, pi/3, -2]), pi) == num_points
    np.random.seed(1)
    points = random_sphere_points(50)
    vector = np.random.rand(3)
    vector /= np.linalg.norm(vector)
    assert count_points_within_alpha(points, vector, pi/6) == 4
    assert count_points_within_alpha(points, vector, pi/3) == 16
    assert count_points_within_alpha(points, vector, pi/2) == 25


def test_random_axes_count_points():
    num_points = 300
    grid_r = build_grid(num_points, "randomQ")
    points = grid_r.get_grid()
    # all points within 180 degrees
    assert np.allclose(random_axes_count_points(points, alpha=pi, num_random_points=300), 1)
    # no points within 0 degrees
    assert np.allclose(random_axes_count_points(points, alpha=0, num_random_points=300), 0)
    # approximately half the points in 90 degrees
    half_circle = random_axes_count_points(points, alpha=pi/2, num_random_points=1000)
    avg = np.average(half_circle)
    assert np.isclose(avg, 0.5, atol=0.01)
    std = np.std(half_circle)
    assert std < 0.05