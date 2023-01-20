from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.constants import pi

from molgri.space.utils import angle_between_vectors, random_sphere_points, random_quaternions, \
    randomise_quaternion_set_signs


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
    # # use half-sphere in both examples
    central_vectors = randomise_quaternion_set_signs(central_vectors)
    array_points = randomise_quaternion_set_signs(array_points)
    all_ratios = np.zeros(num_random_points)
    for i, central_vector in enumerate(central_vectors):
        num_within = count_points_within_alpha(array_points, central_vector, alpha)
        all_ratios[i] = num_within/len(array_points)
    return all_ratios


def prepare_statistics(point_array: NDArray, alphas: list, d=3, num_rand_points=1000) -> Tuple:
    """
    Prepare statistics data that is used for ViolinPlots, either based on 3D points or on quaternions.

    Args:
        point_array: an array of points for which coverage is tested, must be of the shape (N, d)
        alphas: a list of angles in radians at which convergence is tested
        d: number of dimensions (3 or 4)
        num_rand_points: number of random axes around which convergence is tested

    Returns:
        (data_frame_summary, data_frame_full)
    """
    if alphas is None:
        alphas = [pi / 6, 2 * pi / 6, 3 * pi / 6, 4 * pi / 6, 5 * pi / 6]
    assert d == 3 or d == 4, f"Statistics available only for 3 or 4 dimensions, not d={d}"
    assert point_array.shape[1] == d, f"Provided array not in shape (N, {d})"
    if d == 3:
        sphere_surface = 4 * pi
        cone_area_f = lambda a: 2 * pi * (1 - np.cos(a))
        actual_coverages_f = random_axes_count_points
    else:
        # explanation: see https://scialert.net/fulltext/?doi=ajms.2011.66.70&org=11
        sphere_surface = 2 * pi ** 2  # full 4D sphere has area 2pi^2 r^3
        cone_area_f = lambda a: 1 / 2 * sphere_surface * (2 * a - np.sin(2 * a)) / np.pi
        actual_coverages_f = random_quaternions_count_points
    # write out short version ("N points", "min", "max", "average", "SD"
    columns = ["alphas", "ideal coverages", "min coverage", "avg coverage", "max coverage", "standard deviation"]
    ratios_columns = ["coverages", "alphas", "ideal coverage"]
    ratios = [[], [], []]
    data = np.zeros((len(alphas), 6))  # 5 data columns for: alpha, ideal coverage, min, max, average, sd
    for i, alpha in enumerate(alphas):
        cone_area = cone_area_f(alpha)
        ideal_coverage = cone_area / sphere_surface
        actual_coverages = actual_coverages_f(point_array, alpha, num_random_points=num_rand_points)
        ratios[0].extend(actual_coverages)
        ratios[1].extend([alpha] * num_rand_points)
        ratios[2].extend([ideal_coverage] * num_rand_points)
        data[i][0] = alpha
        data[i][1] = ideal_coverage
        data[i][2] = np.min(actual_coverages)
        data[i][3] = np.average(actual_coverages)
        data[i][4] = np.max(actual_coverages)
        data[i][5] = np.std(actual_coverages)
    alpha_df = pd.DataFrame(data=data, columns=columns)
    alpha_df = alpha_df.set_index("alphas")
    ratios_df = pd.DataFrame(data=np.array(ratios).T, columns=ratios_columns)
    return alpha_df, ratios_df


def write_statistics(stat_data: pd.DataFrame, full_data: pd.DataFrame,
                     path_summary: str, path_full_data: str,
                     num_random, name, dimensions, print_message: bool = False):
    # first message (what measure you are using)
    newline = "\n"
    if dimensions == 3:
        m1 = f"STATISTICS: Testing the coverage of grid {name} using {num_random} " \
             f"random points on a sphere. This statistics is useful for the position grid."
    elif dimensions == 4:
        m1 = f"STATISTICS: Testing the coverage of quaternions {name} using {num_random} " \
             f"random quaternions. This statistics is useful for the relative orientations."
    else:
        raise ValueError("Dimensions must be 3 or 4.")
    m2 = f"We select {num_random} random axes and count the number of grid points that fall within the angle" \
         f"alpha (selected from [pi / 6, 2 * pi / 6, 3 * pi / 6, 4 * pi / 6, 5 * pi / 6]) of this axis. For an" \
         f"ideally uniform grid, we expect the ratio of num_within_alpha/total_num_points to equal the ratio" \
         f"area_of_alpha_spherical_cap/area_of_sphere, which we call ideal coverage."
    if print_message:
        print(m1)
        print(stat_data)
    # dealing with the file
    with open(path_summary, "w") as f:
        f.writelines([m1, newline, newline, m2, newline, newline])
    stat_data.to_csv(path_summary, mode="a")
    full_data.to_csv(path_full_data, mode="w")
