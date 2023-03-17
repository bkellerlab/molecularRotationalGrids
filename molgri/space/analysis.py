"""
Analysis of uniformity of 3D and 4D sphere grids.

Always performed by selecting random vectors and evaluating how many grid points fall within the spherical cap area
around this vector. Ideally, the ratio of points within the spherical cap and all points should be equal to the
ratio of sphere cap area and full sphere area. For quaternions, double-coverage must be considered.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.constants import pi

from molgri.space.utils import angle_between_vectors, random_sphere_points, random_quaternions, \
    randomise_quaternion_set_signs
from molgri.constants import DEFAULT_ALPHAS_3D, DEFAULT_ALPHAS_4D, TEXT_ALPHAS_3D, TEXT_ALPHAS_4D


def vector_within_alpha(central_vec: NDArray, side_vec: NDArray, alpha: float) -> NDArray[bool]:
    """
    Calculate the angle between each pair of vectors in zip(central_vec, side_vector). Return an array in which True
    is given where the angle is smaller than alpha and False where not.

    Args:
        central_vec: a single vector of shape (d,) or an array of vectors of shape (n1, d)
        side_vec: a single vector of shape (d,) or an array of vectors of shape (n2, d)
        alpha: in radians, angle around central_vec where we check if side_vectors occur

    Returns:
        array of bools of shape (n1, n2) if both vectors are 2D
                       of shape (n2,) if central_vec is 1D and side_vec 2D
                       of shape (n1,) if central_vec is 2D and side_vec 1D
                       of shape (1,) if both vectors are 1D
    """
    return angle_between_vectors(central_vec, side_vec) < alpha


def count_points_within_alpha(array_points: NDArray, central_vec: NDArray, alpha: float) -> int:
    """
    Count how many points fall within spherical cap with central angle alpha around the central vector.

    Args:
        array_points: each row is a point on a sphere that may or may not be within a spherical cap area
        central_vec: defines the center of spherical cap
        alpha: defines the width of spherical cap

    Returns:
        the number of rows within array_points that are vectors with endings in the spherical cap area.
    """
    # since a list of True or False is returned, the sum gives the total number of True elements
    return int(np.sum(vector_within_alpha(central_vec, array_points, alpha)))


def random_axes_count_points(array_points: NDArray, alpha: float, num_random_points: int = 1000) -> NDArray:
    """
    Repeat count_points_within_alpha for many random selections of the central vector. Using this, we get a good
    estimate of uniformity across the entire sphere.

    Args:
        array_points: each row is a point on a sphere
        alpha: defines the width of spherical cap
        num_random_points: how many random central vectors to test

    Returns:
        Array of shape (num_random_points,) where for each random central vector the ratio
        array points within spherical cal/all array points is saved
    """
    central_vectors = random_sphere_points(num_random_points)
    return _repeat_count(array_points, central_vectors, num_random_points, alpha)


def random_quaternions_count_points(array_points: np.ndarray, alpha: float, num_random_points: int = 1000):
    """
    Same as random_axes_count_points, but because we are dealing with quaternions, random axes are defined differently
    and quaternion signs are randomised first.
    """
    central_vectors = random_quaternions(num_random_points)
    central_vectors = randomise_quaternion_set_signs(central_vectors)
    array_points = randomise_quaternion_set_signs(array_points)
    return _repeat_count(array_points, central_vectors, num_random_points, alpha)


def _repeat_count(array_points, central_vectors, num_random_points, alpha):
    # common function for calculation ratios for vectors/quaternions
    all_ratios = np.zeros(num_random_points)
    for i, central_vector in enumerate(central_vectors):
        num_within = count_points_within_alpha(array_points, central_vector, alpha)
        all_ratios[i] = num_within/len(array_points)
    return all_ratios


def prepare_statistics(point_array: NDArray, alphas: list, d: int = 3, num_rand_points: int = 1000) -> Tuple:
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
        if d == 3:
            alphas = DEFAULT_ALPHAS_3D
        else:
            alphas = DEFAULT_ALPHAS_4D
    assert d == 3 or d == 4, f"Statistics available only for 3 or 4 dimensions, not d={d}"
    assert point_array.shape[1] == d, f"Provided array not in shape (N, {d})"
    if d == 3:
        sphere_surface = 4 * pi

        def cone_area_f(a):
            """Cone area 3D sphere, depending on angle a"""
            return 2 * pi * (1 - np.cos(a))

        actual_coverages_f = random_axes_count_points
    else:
        # explanation: see https://scialert.net/fulltext/?doi=ajms.2011.66.70&org=11
        sphere_surface = 2 * pi ** 2  # full 4D sphere has area 2pi^2 r^3

        def cone_area_f(a):
            """Cone area 4D sphere, depending on angle a"""
            return 1 / 2 * sphere_surface * (2 * a - np.sin(2 * a)) / np.pi

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
                     num_random: int, name: str, dimensions: int, print_message: bool = False):
    """
    The top-level method that writes full statistics and a summary to files. Only deals with messages/files, all
    calculations are outsourced to other functions

    Args:
        stat_data: summary data returned by prepare_statistics function
        full_data: full data returned by prepare_statistics function
        path_summary: where to save summary
        path_full_data: where to save full data
        num_random: number of random orientations when calculating statistics
        name: ID of the system
        dimensions: 3 or 4 dimensions (determine sphere and spherical area formulas)
        print_message: True if a summary message should be printed out
    """
    # first message (what measure you are using)
    newline = "\n"
    if dimensions == 3:
        m1 = f"STATISTICS: Testing the coverage of grid {name} using {num_random} " \
             f"random points on a sphere. This statistics is useful for the position grid."
        labels = TEXT_ALPHAS_3D
    elif dimensions == 4:
        m1 = f"STATISTICS: Testing the coverage of quaternions {name} using {num_random} " \
             f"random quaternions. This statistics is useful for the relative orientations."
        labels = TEXT_ALPHAS_4D
    else:
        raise ValueError("Dimensions must be 3 or 4.")
    m2 = f"We select {num_random} random axes and count the number of grid points that fall within the angle" \
         f"alpha (selected from {labels}) of this axis. For an" \
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
