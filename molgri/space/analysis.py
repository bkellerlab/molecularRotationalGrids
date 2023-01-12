import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi

from molgri.space.utils import angle_between_vectors


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


def vector_within_alpha(central_vec: NDArray, side_vector: NDArray, alpha: float) -> bool or NDArray:
    """
    Answers the question: which angles between both vectors or sets of vectors are within angle angle alpha?

    Args:
        central_vec: a single vector of shape (d,) or an array of vectors of shape (n1, d)
        side_vector: a single vector of shape (d,) or an array of vectors of shape (n2, d)
        alpha: in radians, angle around central_vec where we check if side_vectors occur

    Returns:
        array of bools of shape (n1, n2) if both vectors are 2D
                       of shape (n2,) if central_vec is 1D and side_vec 2D
                       of shape (n1,) if central_vec is 2D and side_vec 1D
                       of shape (1,) if both vectors are 1D
    """
    return angle_between_vectors(central_vec, side_vector) < alpha


def count_points_within_alpha(array_points: np.ndarray, central_vec: np.ndarray, alpha: float):
    # since a list of True or False is returned, the sum gives the total number of True elements
    return np.sum(vector_within_alpha(central_vec, array_points, alpha))


def random_axes_count_points(array_points: np.ndarray, alpha: float, num_random_points: int = 1000):
    central_vectors = random_sphere_points(num_random_points)
    all_ratios = np.zeros(num_random_points)
    for i, central_vector in enumerate(central_vectors):
        num_within = count_points_within_alpha(array_points, central_vector, alpha)
        all_ratios[i] = num_within/len(array_points)
    return all_ratios


def random_quaternions_count_points(array_points: np.ndarray, alpha: float, num_random_points: int = 1000):
    central_vectors = random_quaternions(num_random_points)
    # # use half-sphere in both examples - if 1st component negative, flip the whole vector
    # for i, line in enumerate(central_vectors):
    #     if line[0] < 0:
    #         central_vectors[i, :] = -central_vectors[i, :]
    # for i, line in enumerate(array_points):
    #     if line[0] < 0:
    #         array_points[i, :] = -array_points[i, :]
    all_ratios = np.zeros(num_random_points)
    for i, central_vector in enumerate(central_vectors):
        num_within = count_points_within_alpha(array_points, central_vector, alpha)
        all_ratios[i] = num_within/len(array_points)
    return all_ratios


if __name__ == "__main__":
    from molgri.space.rotobj import build_rotations
    from molgri.constants import GRID_ALGORITHMS

    N = 40

    for algo in GRID_ALGORITHMS[:-1]:
        quats = build_rotations(N, algo=algo).rotations.as_quat()
        ratios = random_quaternions_count_points(quats, alpha=pi/6)
        print(algo, np.average(ratios), np.std(ratios))