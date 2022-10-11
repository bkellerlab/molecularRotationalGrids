import numpy as np
import pandas as pd
from scipy.constants import pi
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .constants import SIZE2NUMBERS
from .paths import PATH_OUTPUT_ROTGRIDS


# TODO: functions here (except unit_dist_on_sphere) are basically not used
# replace them with useful ones that create statistics files

def dist_on_sphere(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    WORKS, but recommend using the equivalent metric="cosine" in cdist because it is faster.

    Find the distance on the surface of a sphere between two vectors that must be pointing to points on a sphere
    with the same number of dimensions and same radius.

    Args:
        vector1: vector to the first point
        vector2: vector to the second point

    Returns:
        distance
    """
    assert vector1.shape == vector2.shape, "Vectors don't have the same shape"
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    assert np.isclose(norm1, norm2), f"Vectors don't have the same norm: {norm1:4f}=/={norm2:4f}."
    radius = norm1
    v1_u = vector1 / norm1
    v2_u = vector2 / norm2
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))  # in radians
    assert angle >= 0
    return radius * angle


def unit_dist_on_sphere(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    Same as dist_on_sphere, but accepts and returns arrays and should only be used for already unitary vectors.

    Args:
        vector1: vector shape (n1, d)
        vector2: vector shape (n2, d)

    Returns:
        an array the shape (n1, n2) containing distances between both sets of points on sphere
    """
    angle = np.arccos(np.clip(np.dot(vector1, vector2.T), -1.0, 1.0))  # in radians
    return angle


def min_max_avg_sd_distance(grid: np.ndarray) -> np.ndarray:
    """
    Samples 1000 random points on a sphere. Finds the distance between each random point and the
    closest grid point. Calculates the the min, max and average distance to nearest grid point over all random points.

    Args:
        grid: an array of shape (N, d) where N is the number of grid points and d the number of dimensions

    Returns:
        array(min_distance, max_distance, average_distance, sd) to closest grid point
    """
    random_points = random_sphere_points(1000)
    distances = cdist(random_points, grid, metric="cosine")
    distances.sort()
    nn_dist = distances[:, 0]
    min_dist = np.min(nn_dist)
    max_dist = np.max(nn_dist)
    average_dist = np.average(nn_dist)
    sd = np.std(nn_dist)
    # optional: can instead express distances as areas of circles on the sphere (around the relevant points)
    # area of a circle on a sphere https://math.stackexchange.com/questions/1832110/area-of-a-circle-on-sphere
    # return 2 * pi * (1 - np.cos(np.array([min_dist, max_dist, average_dist, sd])))
    # or as percentages of sphere surface by additionally / (2*pi) * 100 %
    return np.array([min_dist, max_dist, average_dist, sd])


def coverage_grids(grid_names: list, filename: str = False, redo_calculations: bool = False,
                   set_size: str = "normal") -> pd.DataFrame:
    """
    Pretty redundant. Similar to coverage_per_n(), but specifically for the set sizes 'normal' and 'small'.

    Args:
        grid_names: a list of names to be used
        filename: name of the combined .csv file if you want to save data
        redo_calculations: if True, min_max_avg_sd_distance is recalculated
        set_size: 'normal' or 'small'

    Returns:
        dataframe with "N points", "min", "max", "average", "SD" of the given methods
    """
    if redo_calculations:
        data = np.zeros((len(grid_names), 5))  # 5 data columns for: N, min, max, average, sd
        for i, grid_name in enumerate(tqdm(grid_names)):
            path_to_grid = f"{PATH_OUTPUT_ROTGRIDS}{grid_name}_{SIZE2NUMBERS[set_size][i]}.npy"
            grid = np.load(path_to_grid)
            assert np.allclose(np.linalg.norm(grid, axis=1), 1)
            data[i][0] = len(grid)
            data[i][1:] = min_max_avg_sd_distance(grid)
        df = pd.DataFrame(data, index=grid_names, columns=["N points", "min", "max", "average", "SD"])
        df = df.sort_values("average")
    #     if filename:
    #         df.to_csv(PATH_COMPARE_GRIDS + filename + ".csv")
    # else:
    #     df = pd.read_csv(PATH_COMPARE_GRIDS + filename + ".csv", index_col=0)
    return df


def random_sphere_points(n: int = 1000) -> np.ndarray:
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


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def vector_within_alpha(central_vec: np.ndarray, side_vector: np.ndarray, alpha: float):
    v1_u = unit_vector(central_vec)
    v2_u = unit_vector(side_vector)
    angle_vectors = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle_vectors < alpha


def count_points_within_alpha(grid, central_vec: np.ndarray, alpha: float):
    grid_points = grid.get_grid()
    num_points = 0
    for point in grid_points:
        if vector_within_alpha(central_vec, point, alpha):
            num_points += 1
    return num_points


def random_axes_count_points(grid, alpha: float, num_random_points: int = 1000):
    central_vectors = random_sphere_points(num_random_points)
    all_ratios = np.zeros(num_random_points)

    for i, central_vector in enumerate(central_vectors):
        num_within = count_points_within_alpha(grid, central_vector, alpha)
        all_ratios[i] = num_within/grid.N
    return all_ratios