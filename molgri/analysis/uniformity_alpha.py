import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi
import seaborn as sns
import pandas as pd
from grids.grid import build_grid
from analysis.uniformity_measure import random_sphere_points
from plotting.abstract_multiplot import PanelPlot
from plotting.abstract_plot import AbstractPlot
from my_constants import *


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
        # turn on to enable ideal value labels
        # ticks = self.ax.get_xticks()
        # for i, alpha in enumerate(np.unique(df["alphas"])):
        #     df_fil = df[df["alphas"] == alpha]
        #     self.ax.hlines(y=df_fil["ideal coverage"], xmin=ticks[i]-0.3, xmax=ticks[i]+0.3, colors="black",
        #                    linestyles=":", linewidth=1)
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
            #df["ideal coverage"] = cone_area / sphere_surface
            #df["alphas [deg]"] = np.rad2deg(df["alphas"])
            full_df.append(df)
        full_df = pd.concat(full_df, axis=0, ignore_index=True)
        #label_alphas = [r'$\frac{\pi}{6}$', r'$\frac{2\pi}{6}$', r'$\frac{3\pi}{6}$', r'$\frac{4\pi}{6}$', r'$\frac{5\pi}{6}$']
        sns.lineplot(x=full_df["N"], y=full_df["coverages"], ax=self.ax, hue=full_df["alphas"],
                        palette=color_palette("hls", 5), linewidth=1)
        sns.lineplot(x=full_df["N"], y=full_df["ideal coverage"], style=full_df["alphas"], ax=self.ax, color="black")
        if "ico" in self.data_name:
            self.ax.set_xscale("log")
            self.ax.set_yscale("log")

        #self.ax.set_xlim(np.min(full_df["N"]), np.max(full_df["N"]))
        #self.ax.set_ylim(0, 1)
        self.ax.set_title(NAME2SHORT_NAME[self.parsed_data_name.grid_type])
        self.ax.get_legend().remove()
        # get handles
        #handles, labels = self.ax.get_legend_handles_labels()
        # use them in the legend
        #self.ax.legend(handles, label_alphas, loc='center left', bbox_to_anchor=(1, 0.5))


class PanelAlphaViolinPlot(PanelPlot):

    def __init__(self, data_name, plot_type="panel_alpha"):
        super().__init__(data_name, fig_path=PATH_FIG_TEST, dimensions=2, n_columns=3, n_rows=2,
                         style_type=["talk"], plot_type=plot_type, figsize=(0.7*1.3*DIM_LANDSCAPE[0], 0.7*DIM_LANDSCAPE[0]))

    def create(self, **kwargs):
        super().create(AlphaViolinPlot, plot_type=self.plot_type, style_type=self.style_type,
                       **kwargs)


class PanelAlphaConvergencePlot(PanelPlot):

    def __init__(self, data_name, plot_type="panel_alpha_con"):
        super().__init__(data_name, fig_path=PATH_FIG_TEST, dimensions=2, n_columns=3, n_rows=2,
                         style_type=["talk"], plot_type=plot_type, figsize=(0.7*1.3*DIM_LANDSCAPE[0], 0.7*DIM_LANDSCAPE[0]))

    def create(self, **kwargs):
        super().create(AlphaConvergencePlot, plot_type=self.plot_type, style_type=self.style_type,
                       **kwargs)


if __name__ == "__main__":
    from grids.grid import build_grid
    #alpha = pi/3
    #z_vec = np.array([0, 0, 1])
    #my_grid = build_grid("ico", 50, use_saved=True)
    #random_axes_count_points(my_grid, alpha, num_random_points=50)
    #for name in SIX_METHOD_NAMES:
    #    AlphaConvergencePlot(f"{name}").create()
    #PanelAlphaViolinPlot("600").create()
    PanelAlphaConvergencePlot("").create()
    #AlphaViolinPlot("randomE_50").create()
    #PanelAlphaViolinPlot("50").create()
    #PanelAlphaViolinPlot("100").create()

