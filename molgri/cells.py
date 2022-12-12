"""
This module builds cells that enable space discretisation into cells based on points defined by the Grids.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import SphericalVoronoi
from scipy.constants import pi

from molgri.utils import norm_per_axis, normalise_vectors
from molgri.constants import UNIQUE_TOL, GRID_ALGORITHMS, CELLS_DF_COLUMNS


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
    radius = norms[0, 0]
    assert np.allclose(norms, radius), "All input points must have the same norm."
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
    for plot_num in range(len(algs)): #alg, col in zip(algs, colors):
        N_points = CELLS_DF_COLUMNS[0]
        voranoi_areas = CELLS_DF_COLUMNS[2]
        ideal_areas = CELLS_DF_COLUMNS[3]
        current_ax = ax.ravel()[plot_num]
        voranoi_df = pd.read_csv(f"{PATH_OUTPUT_CELLS}{algs[plot_num]}_{r}_{N_min}_{N_max}.csv")
        sns.lineplot(data=voranoi_df, x=N_points, y=voranoi_areas, errorbar="sd", color=colors[plot_num], ax=current_ax)
        sns.scatterplot(data=voranoi_df, x=N_points, y=voranoi_areas, alpha=0.8, color="black", ax=current_ax, s=1)
        sns.scatterplot(data=voranoi_df, x=N_points, y=ideal_areas, color="black", marker="x")
        # ax[plot_num].set_yscale('log')
        current_ax.set_xscale('log')
        current_ax.set_ylim(0.01, 2)
        current_ax.set_title(f"{NAME2PRETTY_NAME[algs[plot_num]]}")
    plt.savefig(f"{PATH_OUTPUT_PLOTS}areas_{r}_{N_min}_{N_max}.{EXTENSION_FIGURES}", dpi=DEFAULT_DPI)
    plt.close()


if __name__ == "__main__":
    from tqdm import tqdm
    from molgri.grids import build_grid, FullGrid
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from molgri.constants import EXTENSION_FIGURES, NAME2PRETTY_NAME, DEFAULT_DPI, GRID_ALGORITHMS, COLORS
    from molgri.paths import PATH_OUTPUT_PLOTS, PATH_OUTPUT_CELLS

    N_set = [int(i) for i in np.logspace(1, 3, 40)]
    N_set = list(set(N_set))
    N_set.sort()

    algs=GRID_ALGORITHMS[:-1]

    radius = 1

    for plot_num in range(len(algs)): #alg, col in zip(algs, colors):
        all_voranoi_areas = np.zeros((np.sum(N_set), 4))
        current_index = 0

        for i, N in enumerate(tqdm(N_set)):
            pg = FullGrid(b_grid_name="none", o_grid_name=f"{algs[plot_num]}_{N}", t_grid_name=f"[{radius/10}]").get_position_grid()[0]
            try:
                r, voranoi_areas = surface_per_cell_voranoi(pg)
            # miss a point but still save the rest of the data
            except ValueError:
                r, voranoi_areas = np.NaN, np.full(N, np.NaN)
            all_voranoi_areas[current_index:current_index + N, 0] = N
            all_voranoi_areas[current_index:current_index + N, 1] = radius
            all_voranoi_areas[current_index:current_index + N, 2] = voranoi_areas
            all_voranoi_areas[current_index:current_index + N, 3] = surface_per_cell_ideal(N, radius)
            #plt.scatter(N * np.ones(voranoi_areas.shape), voranoi_areas, color="red", marker="o")
            #current_ax.scatter(N, surface_per_cell_ideal(N, r), color="black", marker="x")
            current_index += N
        voranoi_df = pd.DataFrame(data=all_voranoi_areas, columns=CELLS_DF_COLUMNS)
        voranoi_df.to_csv(f"{PATH_OUTPUT_CELLS}{algs[plot_num]}_{int(radius)}_{N_set[0]}_{N_set[-1]}.csv", encoding="utf-8")
    plot_voranoi_convergence(N_min=np.min(N_set), N_max=np.max(N_set), r=int(radius))

