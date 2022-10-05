import numpy as np
from scipy.spatial.distance import cdist


def order_grid_points(grid: np.ndarray, N: int, start_i: int = 1) -> np.ndarray:
    """
    You are provided with a (possibly) unordered grid and return a grid with N points ordered in such a way that
    these N points have the best possible coverage.

    Args:
        grid: grid, array of shape (L, 3) to be ordered where L should be >= N
        N: number of grid points wished at the end
        start_i: from which index to start ordering (in case the first i elements already ordered)

    Returns:
        an array of shape (N, 3) ordered in such a way that these N points have the best possible coverage.
    """
    if N > len(grid):
        print(f"Warning! N>len(grid)! Only {len(grid)} points will be returned!")
    for index in range(start_i, min(len(grid), N)):
        grid = select_next_gridpoint(grid, index)
    return grid[:N]


def select_next_gridpoint(set_grid_points, i):
    """
    Provide a set of grid points where the first i are already sorted. Find the best next gridpoint out of points
    in set_grid_points[i:]

    Args:
        set_grid_points: grid, array of shape (L, 3) where elements up to i are already ordered
        i: index how far the array is already ordered (up to bun not including i).

    Returns:
        set_grid_points where the ith element in swapped with the best possible next grid point
    """
    distances = cdist(set_grid_points[i:], set_grid_points[:i], metric="cosine")
    distances.sort()
    nn_dist = distances[:, 0]
    index_max = np.argmax(nn_dist)
    set_grid_points[[i, i + index_max]] = set_grid_points[[i + index_max, i]]
    return set_grid_points


if __name__ == "__main__":
    ex = np.array([[1, 2, 3], [-3, 1, 7], [2, 3, 4], [-3, -4, -3]])
    ex = np.divide(ex, np.linalg.norm(ex, axis=1).reshape(-1, 1))
    order_grid_points(ex, 4, 1)
    # select_next_gridpoint(ex, 1)
