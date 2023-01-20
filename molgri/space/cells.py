"""
This module builds cells that enable space discretisation into cells based on points defined by the Grids.
"""
from time import time

import numpy as np

from numpy.typing import NDArray
from scipy.spatial import SphericalVoronoi
from scipy.constants import pi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from molgri.space.fullgrid import FullGrid
from molgri.space.utils import norm_per_axis, normalise_vectors
from molgri.constants import CELLS_DF_COLUMNS, EXTENSION_FIGURES, NAME2PRETTY_NAME, DEFAULT_DPI, \
    GRID_ALGORITHMS, COLORS, SMALL_NS, UNIQUE_TOL
from molgri.paths import PATH_OUTPUT_PLOTS, PATH_OUTPUT_CELLS


def surface_per_cell_ideal(N: int, r: float):
    """
    If N points are completely uniformly distributed on a surface of a sphere with radius r, each would cover an
    area of equal size that is simply sphere_surface/N. Units of surface are units of radius squared.

    Args:
        N: number of grid points
        r: radius of the sphere

    Returns:
        surface per point for an idealised grid
    """
    sphere_surface = 4*pi*r**2
    return sphere_surface/N


def surface_per_cell_voranoi(points: NDArray):
    """
    This function requires all points to be on a single sphere with a consistent radius.


    Args:
        points: (N, 3)-dimensional array in which each line is a point on a surface of a sphere

    Returns:
        """
    sv = voranoi_surfaces_on_sphere(points)
    radius = sv.radius
    areas = sv.calculate_areas()
    return radius, areas


def voranoi_surfaces_on_sphere(points: NDArray) -> SphericalVoronoi:
    """
    This function requires all points to be on a single sphere with a consistent radius.


    Args:
        points: (N, 3)-dimensional array in which each line is a point on a surface of a sphere

    Returns:

    """
    assert points.shape[1] == 3, "Must provide an input array of size (N, 3)!"
    norms = norm_per_axis(points)
    radius = np.average(norms)
    assert np.allclose(norms, radius, atol=1, rtol=0.01), "All input points must have the same norm."
    # re-normalise for perfect match
    points = normalise_vectors(points, length=radius)
    return SphericalVoronoi(points, radius=radius, threshold=10**-UNIQUE_TOL)


def voranoi_surfaces_on_stacked_spheres(points: NDArray) -> list:
    """
    This function deals with all points in position grid

    Args:
        points: an array of shape (n_trans, n_rot, 3)

    Returns:

    """
    result = []
    for subarray in points:
        result.append(voranoi_surfaces_on_sphere(subarray))
    return result


def plot_voranoi_convergence(N_min, N_max, r, algs=GRID_ALGORITHMS[:-1]):
    sns.set_context("talk")
    colors = COLORS
    fig, ax = plt.subplots(2, 3, sharex="all", figsize=(20, 16))
    for plot_num in range(len(algs)):
        N_points = CELLS_DF_COLUMNS[0]
        voranoi_areas = CELLS_DF_COLUMNS[2]
        ideal_areas = CELLS_DF_COLUMNS[3]
        current_ax = ax.ravel()[plot_num]
        voranoi_df = pd.read_csv(f"{PATH_OUTPUT_CELLS}{algs[plot_num]}_{r}_{N_min}_{N_max}.csv")
        sns.lineplot(data=voranoi_df, x=N_points, y=voranoi_areas, errorbar="sd", color=colors[plot_num], ax=current_ax)
        sns.scatterplot(data=voranoi_df, x=N_points, y=voranoi_areas, alpha=0.8, color="black", ax=current_ax, s=1)
        sns.scatterplot(data=voranoi_df, x=N_points, y=ideal_areas, color="black", marker="x", ax=current_ax)
        # ax[plot_num].set_yscale('log')
        current_ax.set_xscale('log')
        current_ax.set_ylim(0.01, 2)
        current_ax.set_title(f"{NAME2PRETTY_NAME[algs[plot_num]]}")
    plt.savefig(f"{PATH_OUTPUT_PLOTS}areas_{r}_{N_min}_{N_max}.{EXTENSION_FIGURES}", dpi=DEFAULT_DPI)
    plt.close()


def save_voranoi_data_for_alg(alg_name: str, N_set: tuple = SMALL_NS, radius: float = 1):
    """

    Args:
        alg_name: name of the used algorithm
        N_set: sorted list of points used foor calculating Voranoi areas
        radius: radius of the sphere on which the tesselation takes place

    Returns:

    """
    assert alg_name in GRID_ALGORITHMS, f"Name of the algorithm: {alg_name} is unknown."
    assert radius > 0, f"Radius must be a positive float, unrecognised argument: {radius}."
    all_voranoi_areas = np.zeros((np.sum(N_set), len(CELLS_DF_COLUMNS)))
    current_index = 0

    for i, N in enumerate(N_set):
        t1_grid = time()
        pg = FullGrid(b_grid_name="none", o_grid_name=f"{alg_name}_{N}",
                      t_grid_name=f"[{radius / 10}]").get_position_grid()
        pg = np.swapaxes(pg, 0, 1)[0]
        t2_grid = time()
        grid_time = t2_grid - t1_grid
        try:
            t1 = time()
            r, voranoi_areas = surface_per_cell_voranoi(pg)
            t2 = time()
            tesselation_time = t2 - t1
        # miss a point but still save the rest of the data
        # TODO: deal with duplicate generators issue
        except ValueError:
            r, voranoi_areas = np.NaN, np.full(N, np.NaN)
            tesselation_time = np.NaN
        all_voranoi_areas[current_index:current_index + N, 0] = N
        all_voranoi_areas[current_index:current_index + N, 1] = radius
        all_voranoi_areas[current_index:current_index + N, 2] = voranoi_areas
        all_voranoi_areas[current_index:current_index + N, 3] = surface_per_cell_ideal(N, radius)
        all_voranoi_areas[current_index:current_index + N, 4] = grid_time
        all_voranoi_areas[current_index:current_index + N, 5] = tesselation_time
        current_index += N
    voranoi_df = pd.DataFrame(data=all_voranoi_areas, columns=CELLS_DF_COLUMNS)
    voranoi_df.to_csv(f"{PATH_OUTPUT_CELLS}{alg_name}_{int(radius)}_{N_set[0]}_{N_set[-1]}.csv", encoding="utf-8")
