import numpy as np
from scipy.constants import pi


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


if __name__ == '__main__':
    from .grids import IcoGrid
    my_grid = IcoGrid(1000).get_grid()
    min_radius = 0.5  # nm
    min_alpha = np.inf
    for i in range(1000):
        for j in range(i+1, 1000):
            alpha = unit_dist_on_sphere(my_grid[i], my_grid[j])
            if alpha < min_alpha:
                min_alpha = alpha
    print(min_radius*min_alpha*10, " angstrom")