from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.constants import pi
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib import ticker
from mpl_toolkits.mplot3d.axes3d import Axes3D
from seaborn import color_palette

from molgri.analysis import vector_within_alpha
from molgri.parsers import NameParser, FullGridNameParser
from molgri.utils import norm_per_axis
from .grids import Polytope, IcosahedronPolytope, CubePolytope, build_grid_from_name, FullGrid
from .constants import DIM_SQUARE, DEFAULT_DPI, COLORS, DEFAULT_NS, EXTENSION_FIGURES
from .paths import PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS, PATH_OUTPUT_FULL_GRIDS


class AbstractPlot(ABC):
    """
    Most general plotting class (for one axis per plot). Set all methods that could be useful more than one time here.
    All other plots, including multiplots, inherit from this class.
    """

    def __init__(self, data_name: str, dimensions: int = 3, style_type: list = None, fig_path: str = PATH_OUTPUT_PLOTS,
                 ani_path: str = PATH_OUTPUT_ANIS, ax: Union[Axes, Axes3D] = None, figsize: tuple = DIM_SQUARE,
                 plot_type: str = "abs"):
        """
        Input all information that needs to be provided before fig and ax are created.

        Args:
            data_name: eg ico_500_full
            dimensions: 2 or 3
            style_type: a list of properties like 'dark', 'talk', 'empty' or 'half_empty'
            fig_path: folder to save figures if created, should be set in subclasses
            ani_path: folder to save animations if created, should be set in subclasses
            ax: enables to pass an already created axis - useful for PanelPlots
            figsize: forwarded to set up of the figure
            plot_type: str describing the plot function, added to the name of the plot
        """
        if style_type is None:
            style_type = ["white"]
        self.fig_path = fig_path
        self.ani_path = ani_path
        self.dimensions = dimensions
        assert self.dimensions in [2, 3]
        self.style_type = style_type
        self.data_name = data_name
        self.plot_type = plot_type
        self.figsize = figsize
        # here change styles that need to be set before fig and ax are created
        sns.reset_orig()
        plt.style.use('default')
        if "dark" in style_type:
            plt.style.use('dark_background')
        if "talk" in style_type:
            sns.set_context("talk")
        # create the empty figure
        self.fig = None
        self.ax = ax

    def create(self, *args, equalize=False, neg_limit=None, pos_limit=None, x_label=None, y_label=None, z_label=None,
               title=None, save_fig=True, animate_rot=False, animate_seq=False, sci_limit_min=-4, sci_limit_max=4,
               save_ending=EXTENSION_FIGURES, dpi=600, labelpad=0, pad_inches=0, sharex="all", sharey="all", close_fig=True,
               azim=-60, elev=30, main_ticks_only=False):
        """
        This is the only function the user should call on subclasses. It performs the entire plotting and
        saves the result. It uses all methods in appropriate order with appropriate values for the specific
        plot type we are using. If requested, saves the plot and/or animations.
        """
        self._create_fig_ax(sharex=sharex, sharey=sharey)
        self._set_up_empty()
        if self.dimensions == 3:
            self.ax.view_init(azim=azim, elev=elev)
        if equalize:
            self._equalize_axes(neg_limit=neg_limit, pos_limit=pos_limit)
        if x_label or y_label or z_label:
            self._create_labels(x_label=x_label, y_label=y_label, z_label=z_label, labelpad=labelpad)
        if title:
            self._create_title(title=title)
        self._plot_data()
        self._sci_ticks(sci_limit_min, sci_limit_max)
        if main_ticks_only:
            if self.dimensions == 3:
                for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
                    axis.set_major_locator(ticker.MaxNLocator(integer=True))
            else:
                for axis in [self.ax.xaxis, self.ax.yaxis]:
                    axis.set_major_locator(ticker.MaxNLocator(integer=True))
        if save_fig:
            self._save_plot(save_ending=save_ending, dpi=dpi, pad_inches=pad_inches)
        if close_fig:
            plt.close()
        if animate_rot:
            self.animate_figure_view()

    # noinspection PyUnusedLocal
    def _create_fig_ax(self, sharex: str = "all", sharey: str = "all"):
        """
        The parameters need to stay there to be consistent with AbstractMultiPlot, but are not used.

        Args:
            sharex: if multiplots should share the same x axis
            sharey: if multiplots should share the same y axis
        """
        self.fig = plt.figure(figsize=self.figsize)
        if self.ax is None:
            if self.dimensions == 3:
                self.ax = self.fig.add_subplot(111, projection='3d', computed_zorder=False)
            else:
                self.ax = self.fig.add_subplot(111)

    def _set_up_empty(self):
        """
        Second part of setting up the look of the plot, this time deleting unnecessary properties.
        Keywords to change the look of the plot are taken from the property self.style_type.
        If 'half_empty', remove ticks, if 'empty', also any shading of the background in 3D plots.
        The option 'half_dark' changes background to gray, 'dark' to black.
        """
        if "empty" in self.style_type or "half_empty" in self.style_type:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            if self.dimensions == 3:
                self.ax.set_zticks([])
            if "empty" in self.style_type:
                self.ax.axis('off')
        if "half_dark" in self.style_type:
            color = (0.5, 0.5, 0.5, 0.7)
            self.ax.xaxis.set_pane_color(color)
            self.ax.yaxis.set_pane_color(color)
            self.ax.zaxis.set_pane_color(color)

    def _equalize_axes(self, neg_limit: float = None, pos_limit: float = None):
        """
        Makes x, y, (z) axis equally longs and if limits given, enforces them on all axes.

        Args:
            neg_limit: if set, this will be min x, y, (z) value of the plot
            pos_limit: if set, this will be max x, y, (z) value of the plot - if pos_limit set but neg_limit not,
                       neg_limit is set to -pos_limit
        """
        # because ax.set_aspect('equal') does not work for 3D axes
        if self.dimensions == 3:
            self.ax.set_box_aspect(aspect=[1, 1, 1])
            x_lim, y_lim, z_lim = self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()
            all_ranges = abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]), abs(z_lim[1] - z_lim[0])
            x_middle, y_middle, z_middle = np.mean(x_lim), np.mean(y_lim), np.mean(z_lim)
            plot_range = 0.5 * max(all_ranges)
            self.ax.set_xlim3d([x_middle - plot_range, x_middle + plot_range])
            self.ax.set_ylim3d([y_middle - plot_range, y_middle + plot_range])
            self.ax.set_zlim3d([z_middle - plot_range, z_middle + plot_range])
        else:
            self.ax.set_aspect('equal')
        if pos_limit is not None and neg_limit is None:
            neg_limit = -pos_limit
        if pos_limit and neg_limit:
            self.ax.set_xlim(neg_limit, pos_limit)
            self.ax.set_ylim(neg_limit, pos_limit)
            if self.dimensions == 3:
                self.ax.set_zlim(neg_limit, pos_limit)

    @abstractmethod
    def _prepare_data(self) -> object:
        """
        This function should only be used by the self._plot_data method to obtain the data that we wanna plot.

        Returns:
            dataframe, grid or similar construction
        """

    @abstractmethod
    def _plot_data(self, **kwargs):
        """Here, the plotting is implemented in subclasses."""

    def _create_labels(self, x_label: str = None, y_label: str = None, z_label: str = None, **kwargs):
        if x_label:
            self.ax.set_xlabel(x_label, **kwargs)
        if y_label:
            self.ax.set_ylabel(y_label, **kwargs)
        if z_label and self.dimensions == 3:
            self.ax.set_zlabel(z_label, **kwargs)

    def _create_title(self, title: str):
        if "talk" in self.style_type:
            self.ax.set_title(title, fontsize=15)
        else:
            self.ax.set_title(title)

    @staticmethod
    def _sci_ticks(neg_lim: int = -4, pos_lim: int = 4):
        try:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(neg_lim, pos_lim))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(neg_lim, pos_lim))
        except AttributeError:
            pass

    def animate_figure_view(self) -> FuncAnimation:
        """
        Rotate the 3D figure for 360 degrees around itself and save the animation.
        """
        plt.close()  # this is necessary

        if self.dimensions == 2:
            raise ValueError("Animation of figure rotation only available for 3D figures!")

        def animate(frame):
            # rotate the view left-right
            self.ax.view_init(azim=2*frame)
            plt.pause(.001)
            return self.fig

        anim = FuncAnimation(self.fig, animate, frames=180, interval=50)
        writergif = PillowWriter(fps=10, bitrate=-1)
        # noinspection PyTypeChecker
        anim.save(f"{self.ani_path}{self.data_name}_{self.plot_type}.gif", writer=writergif, dpi=400)
        return anim

    def _save_plot(self, save_ending: str = EXTENSION_FIGURES, dpi: int = DEFAULT_DPI, **kwargs):
        self.fig.tight_layout()
        standard_name = self.data_name
        plt.savefig(f"{self.fig_path}{standard_name}_{self.plot_type}.{save_ending}", dpi=dpi, bbox_inches='tight',
                    **kwargs)
        plt.close()


class GridPlot(AbstractPlot):

    def __init__(self, data_name, *, style_type: list = None, plot_type: str = "grid", **kwargs):
        """
        This class is used for plots and animations of grids.

        Args:
            data_name: in the form algorithm_N e.g. randomQ_60
            style_type: a list of style properties like ['empty', 'talk', 'half_dark']
            plot_type: change this if you need unique name for plots with same data_name
            **kwargs:
        """
        if style_type is None:
            style_type = ["talk"]
        super().__init__(data_name, style_type=style_type, plot_type=plot_type, **kwargs)
        self.grid = self._prepare_data()

    def _prepare_data(self) -> np.ndarray:
        my_grid = build_grid_from_name(self.data_name, use_saved=True).get_grid()
        return my_grid

    def _plot_data(self, **kwargs):
        self.sc = self.ax.scatter(*self.grid.T, color="black", s=30) #, s=4)
        self.ax.view_init(elev=10, azim=30)

    def create(self, **kwargs):
        if "empty" in self.style_type:
            pad_inches = -0.2
        else:
            pad_inches = 0
        pos_limit = kwargs.pop("pos_limit", 1)
        super(GridPlot, self).create(equalize=True, pos_limit=pos_limit, pad_inches=pad_inches, **kwargs)
        animate_seq = kwargs.pop("animate_seq", False)
        if animate_seq:
            self.animate_grid_sequence(pos_lim=pos_limit)

    def animate_grid_sequence(self, pos_lim=1):
        """
        Animate how a grid is constructed - how each individual point is added.

        WARNING - I am not sure that this method always displays correct order/depth coloring - mathplotlib
        is not the most reliable tool for 3d plots and it may change the plotting order for rendering some
        points above others!

        Args:
            pos_lim:
        """

        def update(i):
            current_colors = np.concatenate([facecolors_before[:i], all_white[i:]])
            self.sc.set_facecolors(current_colors)
            self.sc.set_edgecolors(current_colors)
            return self.sc,

        facecolors_before = self.sc.get_facecolors()
        shape_colors = facecolors_before.shape
        all_white = np.zeros(shape_colors)

        self.ax.view_init(elev=10, azim=30)
        self._equalize_axes(pos_limit=pos_lim)
        ani = FuncAnimation(self.fig, func=update, frames=len(facecolors_before), interval=5, repeat=False)
        writergif = PillowWriter(fps=3, bitrate=-1)
        # noinspection PyTypeChecker
        ani.save(f"{self.ani_path}{self.data_name}_{self.plot_type}_ord.gif", writer=writergif, dpi=400)
        plt.close()


class PositionGridPlot(GridPlot):

    def __init__(self, data_name, style_type=None, plot_type="position_grid", **kwargs):
        super().__init__(data_name, style_type=style_type, plot_type=plot_type, **kwargs)

    def _prepare_data(self) -> np.ndarray:
        points = np.load(f"{PATH_OUTPUT_FULL_GRIDS}{self.data_name}.npy")
        return points

    def create(self, **kwargs):
        max_norm = np.max(norm_per_axis(self.grid))
        super(PositionGridPlot, self).create(pos_limit=max_norm, **kwargs)


class GridColoredWithAlphaPlot(GridPlot):
    def __init__(self, data_name, vector: np.ndarray, alpha_set: list, plot_type: str = "colorful_grid", **kwargs):
        super().__init__(data_name, plot_type=plot_type, **kwargs)
        self.alpha_central_vector = vector
        self.alpha_set = alpha_set
        self.alpha_set.sort()
        self.alpha_set.append(pi)

    def _plot_data(self, **kwargs):
        # plot vector
        self.ax.scatter(*self.alpha_central_vector, marker="x", c="k", s=30)
        # determine color palette
        cp = sns.color_palette("Spectral", n_colors=len(self.alpha_set))
        # sort points which point in which alpha area
        already_plotted = []
        for i, alpha in enumerate(self.alpha_set):
            possible_points = np.array([vec for vec in self.grid if tuple(vec) not in already_plotted])
            within_alpha = vector_within_alpha(self.alpha_central_vector, possible_points, alpha)
            selected_points = [tuple(vec) for i, vec in enumerate(possible_points) if within_alpha[i]]
            array_sel_points = np.array(selected_points)
            self.sc = self.ax.scatter(*array_sel_points.T, color=cp[i], s=30)  # , s=4)
            already_plotted.extend(selected_points)
        self.ax.view_init(elev=10, azim=30)


class AlphaViolinPlot(AbstractPlot):

    def __init__(self, data_name: str, *, plot_type: str = "uniformity", style_type: list = None, **kwargs):
        """
        Creates violin plots that are a measure of grid uniformity. A good grid will display minimal variation
        along a range of angles alpha.

        Args:
            data_name: in the form algorithm_N e.g. randomQ_60
            plot_type: change this if you need unique name for plots with same data_name
            style_type: a list of style properties like ['empty', 'talk', 'half_dark']
            **kwargs:
        """
        if style_type is None:
            style_type = ["white"]
        super().__init__(data_name, dimensions=2, style_type=style_type, plot_type=plot_type, **kwargs)

    def _prepare_data(self) -> pd.DataFrame:
        my_grid = build_grid_from_name(self.data_name, use_saved=True)
        # if statistics file already exists, use it, else create it
        try:
            ratios_df = pd.read_csv(my_grid.statistics_path)
        except FileNotFoundError:
            my_grid.save_statistics()
            ratios_df = pd.read_csv(my_grid.statistics_path)
        return ratios_df

    def _plot_data(self, **kwargs):
        df = self._prepare_data()
        sns.violinplot(x=df["alphas"], y=df["coverages"], ax=self.ax, palette=COLORS, linewidth=1, scale="count")
        self.ax.set_xticklabels([r'$\frac{\pi}{6}$', r'$\frac{2\pi}{6}$', r'$\frac{3\pi}{6}$', r'$\frac{4\pi}{6}$',
                                 r'$\frac{5\pi}{6}$'])


class AlphaConvergencePlot(AlphaViolinPlot):

    def __init__(self, data_name: str, **kwargs):
        """
        Creates convergence plots that show how coverages approach optimal values

        Args:
            data_name: name of the algorithm e.g. randomQ
            **kwargs:
        """
        self.nap = NameParser(data_name)
        if self.nap.N is None:
            self.ns_list = np.array(DEFAULT_NS, dtype=int)
        else:
            self.ns_list = np.logspace(np.log10(3), np.log10(self.nap.N), dtype=int)
            self.ns_list = np.unique(self.ns_list)
        super().__init__(data_name, plot_type="convergence", **kwargs)

    def _plot_data(self, **kwargs):
        full_df = []
        for N in self.ns_list:
            self.nap.N = N
            self.data_name = f"{self.nap.algo}_{self.nap.N}"
            df = self._prepare_data()
            df["N"] = N
            full_df.append(df)
        full_df = pd.concat(full_df, axis=0, ignore_index=True)
        sns.lineplot(x=full_df["N"], y=full_df["coverages"], ax=self.ax, hue=full_df["alphas"],
                     palette=color_palette("hls", 5), linewidth=1)
        sns.lineplot(x=full_df["N"], y=full_df["ideal coverage"], style=full_df["alphas"], ax=self.ax, color="black")
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.get_legend().remove()


class PolytopePlot(AbstractPlot):

    def __init__(self, data_name: str, num_divisions=3, faces=None, projection=False, **kwargs):
        """
        Plotting (some faces of) polyhedra, demonstrating the subdivision of faces with points.

        Args:
            data_name:
            num_divisions: how many levels of faces subdivisions should be drawn
            faces: a set of indices indicating which faces to draw
            projection: if True display points projected on a sphere, not on faces
            **kwargs:
        """
        self.num_divisions = num_divisions
        self.faces = faces
        self.projection = projection
        plot_type = f"polytope_{num_divisions}"
        super().__init__(data_name, fig_path=PATH_OUTPUT_PLOTS, plot_type=plot_type, style_type=["empty"], dimensions=3,
                         **kwargs)

    def _prepare_data(self) -> Polytope:
        if self.data_name == "ico":
            ico = IcosahedronPolytope()
        else:
            ico = CubePolytope()
        for n in range(self.num_divisions):
            ico.divide_edges()
        return ico

    def _plot_data(self, **kwargs):
        if self.faces is None and self.data_name == "ico":
            self.faces = {12}
        elif self.faces is None:
            self.faces = {3}
        ico = self._prepare_data()
        ico.plot_points(self.ax, select_faces=self.faces, projection=self.projection)
        ico.plot_edges(self.ax, select_faces=self.faces)


if __name__ == "__main__":
    PositionGridPlot("position_grid_o_cube3D_9_b_zero_1_t_3203903466").create(animate_rot=True, animate_seq=True)