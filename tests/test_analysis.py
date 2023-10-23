import numpy as np
from scipy.constants import pi

from molgri.space.utils import random_sphere_points
from molgri.space.analysis import vector_within_alpha, count_points_within_alpha, random_axes_count_points
from molgri.space.rotobj import SphereGridFactory


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
    grid_r = SphereGridFactory.create(alg_name="randomS", N=num_points, dimensions=3)
    points = grid_r.get_grid_as_array()
    assert count_points_within_alpha(points, np.array([0, 0, 1]), pi) == num_points
    assert count_points_within_alpha(points, np.array([pi/4, pi/3, -2]), pi) == num_points
    np.random.seed(1)
    N = 500
    points = random_sphere_points(N)
    vector = np.random.rand(3)
    vector /= np.linalg.norm(vector)
    # correct proportion of points at different alphas
    for alpha in (pi/6, pi/3, pi/2, 14*pi/17):
        expected_num = 2 * pi * (1 - np.cos(alpha)) / (4 * pi) * N
        counted_num = count_points_within_alpha(points, vector, alpha)
        assert 0.95 * expected_num < counted_num < 1.05 * expected_num


def test_random_axes_count_points():
    num_points = 300
    grid_r = SphereGridFactory.create(alg_name="randomS", N=num_points, dimensions=3)
    points = grid_r.get_grid_as_array()
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


if __name__ == "__main__":
    test_vector_within_alpha()
    test_count_within_alpha()
    test_random_axes_count_points()