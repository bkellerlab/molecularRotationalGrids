"""
Cube_3D grid is outdated! But still implements cube_4D.
"""

import numpy as np


def classic_grid_d_cube(n: int, d: int, cheb: bool = False, change_start: float = 0,
                        change_end: float = 0, dtype=np.float64) -> np.ndarray:
    """
    This is a function to create a classical grid of a d-dimensional cube. It creates a grid over the entire
    (hyper)volume of the (hyper)cube.

    This is a unit cube between -sqrt(1/d) and sqrt(1/d) in all dimensions where d = num of dimensions.

    Args:
        n: number of grid points in each dimension
        d: number of dimensions of the cube
        cheb: use Chebyscheff points instead of equally spaced points
        change_start: add or subtract from -sqrt(1/d) as the start of the grid
        change_end: add or subtract from sqrt(1/d) as the end of the grid
        dtype: forwarded to linspace while creating a grid

    Returns:
        a meshgrid of dimension (d, n, n, .... n) where n is repeated d times
    """
    if cheb:
        from numpy.polynomial.chebyshev import chebpts1
        side = chebpts1(n)
    else:
        # np.sqrt(1/d)
        side = np.linspace(-1 + change_start, 1 - change_end, n, dtype=dtype)
    # repeat the same n points d times and then make a new line of the array every d elements
    sides = np.tile(side, d)
    sides = sides[np.newaxis, :].reshape((d, n))
    # create a grid by meshing every line of the sides array
    return np.array(np.meshgrid(*sides))


def sukharev_grid_d_cube(n: int, d: int) -> np.ndarray:
    """
    This is almost like the classic grid, just points at mid-cells instead of at the edges
    NOT TESTED
    """
    step = 2*np.sqrt(1/d) / n
    return classic_grid_d_cube(n, d, change_start=step/2, change_end=-step/2)


def select_only_faces(grid: np.ndarray) -> np.ndarray:
    """
    Take a meshgrid (d, n, n, ... n)  and return an array of points (N, d) including only the points that
    lie on the faces of the grid, so the edge points in at least one of dimensions.

    Args:
        grid: numpy array (d, n, n, ... n) containing grid points

    Returns:
        points (N, d) where N is the number of edge points and d the dimension
    """
    d = len(grid)
    set_grids = []
    for swap_i in range(d):
        meshgrid_swapped = np.swapaxes(grid, axis1=1, axis2=(1+swap_i))
        set_grids.append(meshgrid_swapped[:, 0, ...])
        set_grids.append(meshgrid_swapped[:, -1, ...])

    result = np.hstack(set_grids).reshape((d, -1)).T
    return np.unique(result, axis=0)


def project_grid_on_sphere(grid: np.ndarray) -> np.ndarray:
    """
    A grid can be seen as a collection of vectors to gridpoints. If a vector is scaled to 1, it will represent a point
    on a unit sphere in d-1 dimensions. This function normalizes the vectors, creating vectors pointing to the
    surface of a d-1 dimensional sphere.

    Args:
        grid: a (N, d) array where each row represents the coordinates of a grid point

    Returns:
        a (N, d) array where each row has been scaled to length 1
    """
    largest_abs = np.max(np.abs(grid), axis=1)[:, np.newaxis]
    grid = np.divide(grid, largest_abs)
    norms = np.linalg.norm(grid, axis=1)[:, np.newaxis]
    return np.divide(grid, norms)


def select_half_sphere(grid: np.ndarray) -> np.ndarray:
    """
    In order to sample SO(3) rotations, we only sample half of the sphere. This function takes a grid and returns
    the same grid with only positive values for the last dimension

    Args:
        grid: an array of points, shape (N, d)

    Returns:
        an array of points, shape (N//2, d), first components of each row >= 0)
    """
    d = grid.shape[1]
    grid_pos = grid[grid[:, d-1] >= 0, :]
    return grid_pos


def cube_grid_on_sphere(n: int, d: int = 3) -> np.ndarray:
    """
    Note: n is not the end number of points, but the number of points per side. For 3D cubes, the final number of
    points can be calculated as: n*n*6 - 12*n + 8, so for example:
    n=2, d=3 -> return array of shape (8, 3)
    n=3, d=3 -> return array of shape (26, 3)
    n=4, d=3 -> return array of shape (56, 3)

    Args:
        n: number of grid points in each dimension
        d: number of dimensions of the cube

    Returns:
        an array of points on a surface of a d-dim sphere, each line one point
    """
    grid_qua = classic_grid_d_cube(n, d)
    grid_qua = select_only_faces(grid_qua)
    grid_qua = project_grid_on_sphere(grid_qua)
    return grid_qua


if __name__ == "__main__":
    my_N = 4
    my_D = 4
    cube_grid_on_sphere(my_N, my_D)
    my_N = 6
