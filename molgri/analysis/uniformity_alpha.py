import numpy as np
from scipy.constants import pi
import seaborn as sns
import pandas as pd

from ..grids.grid import build_grid
from ..analysis.uniformity_measure import random_sphere_points
from ..plotting.abstract_plot import AbstractPlot
from ..my_constants import *


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


class AlphaViolinPlot(AbstractPlot):

    def __init__(self, data_name: str, plot_type="alpha", style_type=None, **kwargs):
        if style_type is None:
            style_type = ["talk"]
        super().__init__(data_name, dimensions=2, style_type=style_type, plot_type=plot_type, fig_path=PATH_FIG_TEST,
                         **kwargs)

    def _prepare_data(self) -> pd.DataFrame:
        my_grid = build_grid(self.parsed_data_name.grid_type, self.parsed_data_name.num_grid_points, use_saved=True)
        alphas = [pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6]
        ratios = [[], [], []]
        num_rand_points = 100
        sphere_surface = 4 * pi
        for alpha in alphas:
            cone_area = 2 * pi * (1-np.cos(alpha))
            ideal_coverage = cone_area / sphere_surface
            ratios[0].extend(random_axes_count_points(my_grid, alpha, num_random_points=num_rand_points))
            ratios[1].extend([alpha]*num_rand_points)
            ratios[2].extend([ideal_coverage]*num_rand_points)
        alpha_df = pd.DataFrame(data=np.array(ratios).T, columns=["coverages", "alphas", "ideal coverage"])
        return alpha_df

    def _plot_data(self, **kwargs):
        df = self._prepare_data()
        sns.violinplot(x=df["alphas"], y=df["coverages"], ax=self.ax, palette=COLORS, linewidth=1, scale="count")
        self.ax.set_title(NAME2SHORT_NAME[self.parsed_data_name.grid_type])
        self.ax.set_xticklabels([r'$\frac{\pi}{6}$', r'$\frac{2\pi}{6}$', r'$\frac{3\pi}{6}$', r'$\frac{4\pi}{6}$',
                                 r'$\frac{5\pi}{6}$'])


class AlphaConvergencePlot(AlphaViolinPlot):

    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, **kwargs)

    def _plot_data(self, **kwargs):
        full_df = []

        for N in DEFAULT_NS:
            self.parsed_data_name.num_grid_points = N
            self.data_name = self.parsed_data_name.get_standard_name()
            df = self._prepare_data()
            df["N"] = N
            full_df.append(df)
        full_df = pd.concat(full_df, axis=0, ignore_index=True)
        sns.lineplot(x=full_df["N"], y=full_df["coverages"], ax=self.ax, hue=full_df["alphas"],
                        palette=color_palette("hls", 5), linewidth=1)
        sns.lineplot(x=full_df["N"], y=full_df["ideal coverage"], style=full_df["alphas"], ax=self.ax, color="black")
        if "ico" in self.data_name:
            self.ax.set_xscale("log")
            self.ax.set_yscale("log")
        self.ax.set_title(NAME2SHORT_NAME[self.parsed_data_name.grid_type])
        self.ax.get_legend().remove()

