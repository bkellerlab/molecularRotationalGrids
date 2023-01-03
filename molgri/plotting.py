from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import matplotlib
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.constants import pi
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib import ticker
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial import geometric_slerp
from seaborn import color_palette

from molgri.analysis import vector_within_alpha
from molgri.cells import voranoi_surfaces_on_stacked_spheres, voranoi_surfaces_on_sphere
from molgri.parsers import NameParser, XVGParser, PtParser, GridNameParser, FullGridNameParser
from molgri.utils import norm_per_axis, normalise_vectors, cart2sphA
from .grids import Polytope, IcosahedronPolytope, CubePolytope, build_grid_from_name
from .constants import DIM_SQUARE, DEFAULT_DPI, COLORS, DEFAULT_NS, EXTENSION_FIGURES, CELLS_DF_COLUMNS, \
    FULL_GRID_ALG_NAMES, UNIQUE_TOL, DIM_LANDSCAPE
from .paths import PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS, PATH_OUTPUT_FULL_GRIDS, PATH_OUTPUT_CELLS, PATH_INPUT_ENERGIES, \
    PATH_INPUT_BASEGRO, PATH_OUTPUT_PT


class AbstractPlot(ABC):
    """
    Most general plotting class (for one axis per plot). Set all methods that could be useful more than one time here.
    All other plots, including multiplots, inherit from this class.
    """

    def __init__(self, data_name: str, dimensions: int = 2, style_type: list = None, fig_path: str = PATH_OUTPUT_PLOTS,
                 ani_path: str = PATH_OUTPUT_ANIS, figsize: tuple = DIM_SQUARE,
                 plot_type: str = "abs"):
        """
        Input all information that needs to be provided before fig and ax are created.

        Args:
            data_name: eg ico_500_full
            dimensions: 2 or 3
            style_type: a list of properties like 'dark', 'talk', 'empty' or 'half_empty'
            fig_path: folder to save figures if created, should be set in subclasses
            ani_path: folder to save animations if created, should be set in subclasses
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
        self.ax = None

    def create(self, *args, equalize=False, x_min_limit=None, x_max_limit=None, y_min_limit=None, y_max_limit=None,
               z_min_limit=None, z_max_limit=None, x_label=None, y_label=None, z_label=None,
               title=None, animate_rot=False, animate_seq=False, sci_limit_min=-4, sci_limit_max=4, color="black",
               labelpad=0, sharex="all", sharey="all", main_ticks_only=False, ax: Union[Axes, Axes3D] = None,
               projection=None, **kwargs):
        """
        This is the only function the user should call on subclasses. It performs the entire plotting and
        saves the result. It uses all methods in appropriate order with appropriate values for the specific
        plot type we are using. If requested, saves the plot and/or animations.
        """
        self._create_fig_ax(ax=ax, sharex=sharex, sharey=sharey, projection=projection)
        self._set_up_empty()
        if any([x_min_limit, x_max_limit, y_min_limit, y_max_limit, z_min_limit, z_max_limit]):
            self._axis_limits(x_min_limit=x_min_limit, x_max_limit=x_max_limit,
                              y_min_limit=y_min_limit, y_max_limit=y_max_limit,
                              z_min_limit=z_min_limit, z_max_limit=z_max_limit)
        if x_label or y_label or z_label:
            self._create_labels(x_label=x_label, y_label=y_label, z_label=z_label, labelpad=labelpad)
        if title:
            self._create_title(title=title)
        self._plot_data(color=color)
        if equalize:
            self._equalize_axes()
        self._sci_ticks(sci_limit_min, sci_limit_max)
        if main_ticks_only:
            self._main_ticks()

    def create_and_save(self, save_ending=EXTENSION_FIGURES, dpi=DEFAULT_DPI, close_fig=True, pad_inches=0, **kwargs):
        self.create(**kwargs)
        self.save_plot(save_ending=save_ending, dpi=dpi, pad_inches=pad_inches)
        if close_fig:
            plt.close()

    def _main_ticks(self):
        for axis in [self.ax.xaxis, self.ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

    def _axis_limits(self, x_min_limit=None, x_max_limit=None, y_min_limit=None, y_max_limit=None, **kwargs):
        if x_max_limit is not None and x_min_limit is None:
            x_min_limit = - x_max_limit
        self.ax.set_xlim(x_min_limit, x_max_limit)
        if y_max_limit is not None and y_min_limit is None:
            y_min_limit = - y_max_limit
        self.ax.set_ylim(y_min_limit, y_max_limit)

    # noinspection PyUnusedLocal
    def _create_fig_ax(self, ax: Union[Axes, Axes3D], sharex: str = "all", sharey: str = "all", projection=None):
        """
        The parameters need to stay there to be consistent with AbstractMultiPlot, but are not used.

        Args:
            sharex: if multiplots should share the same x axis
            sharey: if multiplots should share the same y axis
        """
        self.fig = plt.figure(figsize=self.figsize)
        if ax is None:
            if self.dimensions == 2:
                self.ax = self.fig.add_subplot(111)
            elif self.dimensions == 3 or projection is not None:
                self.ax = self.fig.add_subplot(111, projection=projection) #, computed_zorder=False
            else:
                raise ValueError(f"Dimensions must be in (2, 3), unknown value {self.dimensions}!")
        else:
            self.ax = ax

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

            if "empty" in self.style_type:
                self.ax.axis('off')
        color = (0.5, 0.5, 0.5, 0.7)
        if "half_dark" in self.style_type:
            self.ax.xaxis.set_pane_color(color)
            self.ax.yaxis.set_pane_color(color)
        return color

    def _equalize_axes(self):
        """
        Makes x, y, (z) axis equally longs and if limits given, enforces them on all axes.
        """
        self.ax.set_aspect('equal')

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

    def _create_labels(self, x_label: str = None, y_label: str = None, **kwargs):
        if x_label:
            self.ax.set_xlabel(x_label, **kwargs)
        if y_label:
            self.ax.set_ylabel(y_label, **kwargs)

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

    def save_plot(self, save_ending: str = EXTENSION_FIGURES, dpi: int = DEFAULT_DPI, **kwargs):
        self.fig.tight_layout()
        standard_name = self.data_name
        plt.savefig(f"{self.fig_path}{standard_name}_{self.plot_type}.{save_ending}", dpi=dpi, bbox_inches='tight',
                    **kwargs)
        plt.close()


class AbstractMultiPlot(ABC):

    def __init__(self, list_plots: List[AbstractPlot], figsize=None, n_rows: int = 4, n_columns: int = 2,
                 plot_type: str = "multi"):
        self.ani_path = PATH_OUTPUT_ANIS
        self.dimensions = list_plots[0].dimensions
        assert np.all([list_plot.dimensions == self.dimensions for list_plot in list_plots]), "Plots cannot have various" \
                                                                                              "numbers of directions"
        self.n_rows = n_rows
        self.n_columns = n_columns
        assert self.n_rows*self.n_columns == len(list_plots), f"Specify the number of rows and columns that corresponds" \
                                                              f"to the number of the elements! " \
                                                              f"{self.n_rows}x{self.n_columns}=/={len(list_plots)}"
        self.figsize = figsize
        if self.figsize is None:
            self.figsize = self.n_columns * DIM_SQUARE[0], self.n_rows * DIM_SQUARE[1]

        self.list_plots = list_plots
        self.plot_type = plot_type

    def _create_fig_ax(self, sharex="all", sharey="all", projection=None):
        if self.dimensions == 3:
            # for 3D plots it makes no sense to share axes
            self.fig, self.all_ax = plt.subplots(self.n_rows, self.n_columns, subplot_kw={'projection': projection},
                                                 figsize=self.figsize)
        elif self.dimensions == 2:
            self.fig, self.all_ax = plt.subplots(self.n_rows, self.n_columns, sharex=sharex, sharey=sharey,
                                                 figsize=self.figsize)
        else:
            raise ValueError("Only 2 or 3 dimensions possible.")

    def _remove_midlabels(self):
        if self.dimensions == 2:
            if self.n_rows > 1 and self.n_columns > 1:
                for my_a in self.all_ax[:, 1:]:
                    for subax in my_a:
                        subax.set_ylabel("")
                for my_a in self.all_ax[:-1, :]:
                    for subax in my_a:
                        subax.set_xlabel("")
            else:
                for my_a in self.all_ax[1:]:
                        my_a.set_ylabel("")

    def unify_axis_limits(self, x_ax = False, y_ax = False, z_ax = False):
        x_lim = [np.infty, -np.infty]
        y_lim = [np.infty, -np.infty]
        z_lim = [np.infty, -np.infty]
        # determine the max and min values
        for i, subaxis in enumerate(self.all_ax.ravel()):
            if subaxis.get_xlim()[0] < x_lim[0]:
                x_lim[0] = subaxis.get_xlim()[0]
            if subaxis.get_xlim()[1] > x_lim[1]:
                x_lim[1] = subaxis.get_xlim()[1]
            if subaxis.get_ylim()[0] < y_lim[0]:
                y_lim[0] = subaxis.get_ylim()[0]
            if subaxis.get_ylim()[1] > y_lim[1]:
                y_lim[1] = subaxis.get_ylim()[1]
            if self.dimensions == 3 and subaxis.get_zlim()[0] < z_lim[0]:
                z_lim[0] = subaxis.get_zlim()[0]
            if self.dimensions == 3 and subaxis.z_lim[1] > z_lim[1]:
                z_lim[1] = subaxis.get_zlim()[1]
        # change them all
        for i, subaxis in enumerate(self.all_ax.ravel()):
            if x_ax:
                subaxis.set_xlim(*x_lim)
            if y_ax:
                subaxis.set_ylim(*y_lim)
            if self.dimensions == 3 and z_ax:
                subaxis.set_zlim(*z_lim)

    def animate_figure_view(self) -> FuncAnimation:
        """
        Rotate all 3D figures for 360 degrees around themselves and save the animation.
        """

        def animate(frame):
            # rotate the view left-right
            for ax in self.all_ax.ravel():
                ax.view_init(azim=2*frame)
            return self.fig

        anim = FuncAnimation(self.fig, animate, frames=180, interval=50)
        writergif = PillowWriter(fps=10, bitrate=-1)
        # noinspection PyTypeChecker
        anim.save(f"{self.ani_path}{self.plot_type}.gif", writer=writergif, dpi=400)
        return anim

    # def add_colorbar(self, cbar_kwargs):
    #     #orientation = cbar_kwargs.pop("orientation", "horizontal")
    #     #cbar_label = cbar_kwargs.pop("cbar_label", None)
    #     # for plot in self.list_plots:
    #     #     cmap = plot.cmap
    #     # print(images)
    #     # cbar = self.fig.colorbar()
    #     # norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    #     # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #     # sm.set_array([])
    #     # self.fig.colorbar(sm, ticks=np.linspace(0, 2, 10),
    #     #              boundaries=np.arange(-0.05, 2.1, .1), ax=self.all_ax.ravel().tolist())
    #     # if orientation == "horizontal":
    #     #     pos = self.all_ax.ravel()[-1].get_position()
    #     #     cax = self.fig.add_axes([pos.x0 -0.05, pos.y0 -0.1, 0.3, 0.03])
    #     #
    #     # else:
    #     #     pos = self.all_ax.ravel()[-1].get_position()
    #     #     cax = self.fig.add_axes([pos.x1 - 1, pos.y0 + 0.0, 0.03, 0.4])
    #     # cbar = self.fig.colorbar(p, orientation=orientation, pad=0.4, cax=cax)
    #     # if orientation == "vertical":
    #     #
    #     #     cax.yaxis.set_label_position('left')
    #     # if cbar_label:
    #     #     cbar.set_label(cbar_label)

    def create(self, *args, rm_midlabels=True, palette=COLORS, titles=None, unify_x=False, unify_y=False,
               unify_z=False, animate_rot=False, joint_cbar=False, cbar_kwargs=None, projection=None, **kwargs):
        self._create_fig_ax(projection=projection)
        if titles:
            assert len(titles) == len(self.list_plots), "Wrong number of titles provided"
        else:
            titles = [None]*len(self.list_plots)
        # catch variables connected with saving and redirect them to saving MultiPlot
        for i, subaxis in enumerate(self.all_ax.ravel()):
            self.list_plots[i].create(*args, ax=subaxis, color=palette[i], title=titles[i], **kwargs)
        if rm_midlabels:
            self._remove_midlabels()
        if np.any([unify_x, unify_y, unify_z]):
            self.unify_axis_limits(unify_x, unify_y, unify_z)
        # if joint_cbar:
        #     self.add_colorbar(cbar_kwargs)
        if animate_rot:
            if self.dimensions == 2:
                print("Warning! Rotation of plots only possible for 3D plots.")
            else:
                self.animate_figure_view()

    def create_and_save(self, *args, close_fig=True, **kwargs):
        self.create(*args, **kwargs)
        self.save_multiplot()
        if close_fig:
            plt.close()

    def save_multiplot(self, save_ending: str = EXTENSION_FIGURES, dpi: int = DEFAULT_DPI, **kwargs):
        self.fig.tight_layout()
        fig_path = self.list_plots[0].fig_path
        standard_name = self.list_plots[-1].data_name
        self.fig.savefig(f"{fig_path}multi_{standard_name}_{self.plot_type}.{save_ending}", dpi=dpi, bbox_inches='tight',
                    **kwargs)


class PanelPlot(AbstractMultiPlot):

    def __init__(self, list_plots: List[AbstractPlot], landscape=True, plot_type="panel", figsize=None):
        if landscape:
            n_rows = 2
            n_columns = 3
        else:
            n_rows = 3
            n_columns = 2
        super().__init__(list_plots, n_columns=n_columns, n_rows=n_rows, plot_type=plot_type, figsize=figsize)

    def create(self, *args, rm_midlabels=True, palette=COLORS, titles=None, unify_x=True, unify_y=True,
               unify_z=True, **kwargs):
        titles = FULL_GRID_ALG_NAMES[:-1]
        super().create(*args, rm_midlabels=rm_midlabels, palette=palette, titles=titles, unify_x=unify_x,
                       unify_y=unify_y, unify_z=unify_z, **kwargs)


class Plot3D(AbstractPlot, ABC):

    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, dimensions=3, **kwargs)

    def create(self, *args, animate_rot=False, animate_seq=False, azim=-60, elev=30, projection="3d", **kwargs):
        super().create(*args, projection=projection, **kwargs)
        self.ax.view_init(azim=azim, elev=elev)
        if animate_rot:
            self.animate_figure_view()

    def _main_ticks(self):
        super()._main_ticks()
        self.ax.zaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    def _axis_limits(self, x_min_limit=None, x_max_limit=None, y_min_limit=None, y_max_limit=None,
                     z_min_limit=None, z_max_limit=None):
        super()._axis_limits(x_min_limit=x_min_limit, x_max_limit=x_max_limit, y_min_limit=y_min_limit,
                             y_max_limit=y_max_limit)
        if z_max_limit is not None and z_min_limit is None:
            z_min_limit = - z_max_limit
        self.ax.set_zlim(z_min_limit, z_max_limit)

    def _set_up_empty(self):
        color = super()._set_up_empty()
        if "empty" in self.style_type or "half_empty" in self.style_type:
            self.ax.set_zticks([])
        if "half_dark" in self.style_type:
            self.ax.zaxis.set_pane_color(color)

    def _equalize_axes(self):
        """
        Makes x, y, (z) axis equally longs and if limits given, enforces them on all axes.
        """
        # because ax.set_aspect('equal') does not work for 3D axes
        self.ax.set_box_aspect(aspect=[1, 1, 1])
        x_lim, y_lim, z_lim = self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()
        all_ranges = abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]), abs(z_lim[1] - z_lim[0])
        x_middle, y_middle, z_middle = np.mean(x_lim), np.mean(y_lim), np.mean(z_lim)
        plot_range = 0.5 * max(all_ranges)
        self.ax.set_xlim3d([x_middle - plot_range, x_middle + plot_range])
        self.ax.set_ylim3d([y_middle - plot_range, y_middle + plot_range])
        self.ax.set_zlim3d([z_middle - plot_range, z_middle + plot_range])

    def _create_labels(self, x_label: str = None, y_label: str = None, z_label: str = None, **kwargs):
        super()._create_labels(x_label=x_label, y_label=y_label, **kwargs)
        if z_label:
            self.ax.set_zlabel(z_label, **kwargs)

    def animate_figure_view(self) -> FuncAnimation:
        """
        Rotate the 3D figure for 360 degrees around itself and save the animation.
        """

        def animate(frame):
            # rotate the view left-right
            self.ax.view_init(azim=2*frame)
            return self.fig

        anim = FuncAnimation(self.fig, animate, frames=180, interval=50)
        writergif = PillowWriter(fps=10, bitrate=-1)
        # noinspection PyTypeChecker
        anim.save(f"{self.ani_path}{self.data_name}_{self.plot_type}.gif", writer=writergif, dpi=400)
        return anim


class GridPlot(Plot3D):

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

    def _plot_data(self, color="black", s=30, **kwargs):
        self.sc = self.ax.scatter(*self.grid.T, color=color, s=s) #, s=4)

    def create(self, animate_seq=False, **kwargs):
        if "empty" in self.style_type:
            pad_inches = -0.2
        else:
            pad_inches = 0
        x_max_limit = kwargs.pop("x_max_limit", 1)
        y_max_limit = kwargs.pop("y_max_limit", 1)
        z_max_limit = kwargs.pop("z_max_limit", 1)
        super(GridPlot, self).create(equalize=True, x_max_limit=x_max_limit, y_max_limit=y_max_limit,
                                     z_max_limit=z_max_limit, pad_inches=pad_inches, azim=30, elev=10, projection="3d",
                                     **kwargs)
        if animate_seq:
            self.animate_grid_sequence()

    def animate_grid_sequence(self):
        """
        Animate how a grid is constructed - how each individual point is added.

        WARNING - I am not sure that this method always displays correct order/depth coloring - mathplotlib
        is not the most reliable tool for 3d plots and it may change the plotting order for rendering some
        points above others!
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
        ani = FuncAnimation(self.fig, func=update, frames=len(facecolors_before), interval=5, repeat=False)
        writergif = PillowWriter(fps=3, bitrate=-1)
        # noinspection PyTypeChecker
        ani.save(f"{self.ani_path}{self.data_name}_{self.plot_type}_ord.gif", writer=writergif, dpi=400)


class PositionGridPlot(GridPlot):

    def __init__(self, data_name, style_type=None, cell_lines=False, plot_type="positions", **kwargs):
        self.cell_lines = cell_lines
        super().__init__(data_name, style_type=style_type, plot_type=plot_type, **kwargs)

    def _prepare_data(self) -> np.ndarray:
        points = np.load(f"{PATH_OUTPUT_FULL_GRIDS}{self.data_name}.npy")
        return points

    def _plot_data(self, color="black", **kwargs):
        points = self._prepare_data()
        for i, point_set in enumerate(points):
            self.sc = self.ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], c=color)
        if self.cell_lines:
            self._draw_voranoi_cells(points)

    def create(self, **kwargs):
        max_norm = np.max(norm_per_axis(self.grid))
        super(PositionGridPlot, self).create(x_max_limit=max_norm, y_max_limit=max_norm, z_max_limit=max_norm, **kwargs)

    def _draw_voranoi_cells(self, points):
        svs = voranoi_surfaces_on_stacked_spheres(points)
        for i, sv in enumerate(svs):
            sv.sort_vertices_of_regions()
            t_vals = np.linspace(0, 1, 2000)
            # plot Voronoi vertices
            self.ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')
            # indicate Voronoi regions (as Euclidean polygons)
            for region in sv.regions:
                n = len(region)
                for j in range(n):
                    start = sv.vertices[region][j]
                    end = sv.vertices[region][(j + 1) % n]
                    norm = np.linalg.norm(start)
                    result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)
                    self.ax.plot(norm * result[..., 0], norm * result[..., 1], norm * result[..., 2], c='k')


class TrajectoryEnergyPlot(Plot3D):

    def __init__(self, data_name: str, plot_type="trajectory", plot_points=False, plot_surfaces=True,
                 selected_Ns=None, **kwargs):
        self.energies = None
        self.property = "Trajectory positions"
        self.unit = r'$\AA$'
        self.selected_Ns = selected_Ns
        self.N_index = 0
        self.plot_points = plot_points
        self.plot_surfaces = plot_surfaces
        super().__init__(data_name, plot_type=plot_type, **kwargs)

    def add_energy_information(self, path_xvg_file, property_name="Potential"):
        self.plot_type += "_energies"
        file_parser = XVGParser(path_xvg_file)
        self.property, property_index = file_parser.get_column_index_by_name(column_label=property_name)
        self.unit = file_parser.get_y_unit()
        self.energies = file_parser.all_values[:, property_index]

    def _prepare_data(self) -> pd.DataFrame:
        try:
            split_name = self.data_name.split("_")
            path_m1 = f"{PATH_INPUT_BASEGRO}{split_name[0]}"
            path_m2 = f"{PATH_INPUT_BASEGRO}{split_name[1]}"
            try:
                gnp = FullGridNameParser("_".join(split_name[2:]))
                num_b = gnp.get_num_b_rot()
                num_o = gnp.get_num_o_rot()
            except AttributeError:
                print("Warning! Trajectory name not in standard format. Will not be able to perform convergence tests.")
                num_b = 1
                num_o = 1
        except AttributeError:
            raise ValueError("Cannot use the name of the XVG file to find the corresponding trajectory. "
                             "Please rename the XVG file to the same name as the XTC file.")
        path_topology = f"{PATH_OUTPUT_PT}{self.data_name}.gro"
        path_trajectory = f"{PATH_OUTPUT_PT}{self.data_name}.xtc"
        my_parser = PtParser(path_m1, path_m2, path_topology, path_trajectory)
        my_data = []
        for i, molecules in enumerate(my_parser.generate_frame_as_molecule()):
            mol1, mol2 = molecules
            com = mol2.get_center_of_mass()
            try:
                current_E = self.energies[i]
            except TypeError:
                current_E = 0
            my_data.append([*np.round(com), current_E])
        my_df = pd.DataFrame(my_data, columns=["x", "y", "z", f"{self.property} {self.unit}"])
        # if multiple shells present, warn and only use the closest one.
        num_t = len(my_df) // num_b // num_o
        if num_t != 1:
            print("Warning! The pseudotrajectory has multiple shells/translation distances. 2D/3D visualisation is "
                  "most suitable for single-shell visualisations. Only the data from the first shell will be used "
                  "in the visualisation.")
            my_df = get_only_one_translation_distance(my_df, num_t)
        # of all points with the same position, select only the orientation with the smallest energy
        df_extract = groupby_min_body_energy(my_df, target_column=f"{self.property} {self.unit}", N_b=num_b)
        # now potentially reduce the number of orientations tested
        self.selected_Ns = test_or_create_Ns(num_o, self.selected_Ns)
        points_up_to_Ns(df_extract, self.selected_Ns, target_column=f"{self.property} {self.unit}")
        return df_extract

    def _plot_data(self, **kwargs):
        my_df = self._prepare_data()
        # determine min and max of the color dimension
        all_positions = my_df[my_df[f"{self.selected_Ns[self.N_index]}"].notnull()][["x", "y", "z"]].to_numpy()
        all_energies = my_df[my_df[f"{self.selected_Ns[self.N_index]}"].notnull()][f"{self.selected_Ns[self.N_index]}"].to_numpy()
        # sort out what not unique
        _, indices = np.unique(all_positions.round(UNIQUE_TOL), axis=0, return_index=True)
        all_positions = np.array([all_positions[index] for index in sorted(indices)])
        all_energies = np.array([all_energies[index] for index in sorted(indices)])
        # TODO: enable colorbar even if not plotting points
        if self.energies is None:
            self.ax.scatter(*all_positions.T, c="black")
        else:
            if self.plot_surfaces:
                try:
                    self._draw_voranoi_cells(all_positions, all_energies)
                except AssertionError:
                    print("Warning! Sperichal Voranoi cells plot could not be produced. Likely all points are "
                          "not at the same radius. Will create a scatterplot instead.")
                    self.plot_points = True
            cmap = ListedColormap((sns.color_palette("coolwarm", 256).as_hex()))
            im = self.ax.scatter(*all_positions.T, c=all_energies, cmap=cmap)
            cbar = self.fig.colorbar(im, ax=self.ax)
            cbar.set_label(f"{self.property} {self.unit}")
            if not self.plot_points:
                im.set_visible(False)
            self.ax.set_title(r"$N_{rot}$ " + f"= {self.selected_Ns[self.N_index]}")

    def _draw_voranoi_cells(self, points, colors):
        sv = voranoi_surfaces_on_sphere(points)
        norm = matplotlib.colors.Normalize(vmin=colors.min(), vmax=colors.max())
        cmap = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        cmap.set_array([])
        fcolors = cmap.to_rgba(colors)
        sv.sort_vertices_of_regions()
        for n in range(0, len(sv.regions)):
            region = sv.regions[n]
            polygon = Poly3DCollection([sv.vertices[region]], alpha=1)
            polygon.set_color(fcolors[n])
            self.ax.add_collection3d(polygon)

    def create(self, *args, title=None, **kwargs):
        if title is None:
            title = f"{self.property} {self.unit}"
        super().create(*args, equalize=True, title=title, **kwargs)


def create_trajectory_energy_multiplot(data_name, Ns=None, animate_rot=False):
    list_single_plots = []
    max_index = 5 if Ns is None else len(Ns)
    for i in range(max_index):
        tep = TrajectoryEnergyPlot(data_name, plot_points=False, plot_surfaces=True, selected_Ns=Ns)
        tep.N_index = i
        tep.add_energy_information(f"{PATH_INPUT_ENERGIES}{data_name}.xvg")
        list_single_plots.append(tep)
    TrajectoryEnergyMultiPlot(list_single_plots, n_columns=max_index, n_rows=1).create_and_save(animate_rot=animate_rot)


class HammerProjectionTrajectory(TrajectoryEnergyPlot, AbstractPlot):

    def __init__(self, data_name: str, plot_type="hammer", figsize=DIM_LANDSCAPE, **kwargs):
        super().__init__(data_name, plot_type=plot_type, plot_surfaces=False, plot_points=True, figsize=figsize,
                         **kwargs)

    def _plot_data(self, **kwargs):

        my_df = self._prepare_data()
        # determine min and max of the color dimension
        # only_index = my_df[my_df[f"{self.selected_Ns[self.N_index]}"].notnull()].index.to_list()
        # all_positions = np.array([*only_index])
        # all_energies = my_df[my_df[f"{self.selected_Ns[self.N_index]}"].notnull()][f"{self.selected_Ns[self.N_index]}"]
        all_positions = my_df[my_df[f"{self.selected_Ns[self.N_index]}"].notnull()][["x", "y", "z"]].to_numpy()
        all_energies = my_df[my_df[f"{self.selected_Ns[self.N_index]}"].notnull()][f"{self.selected_Ns[self.N_index]}"].to_numpy()
        # sort out what not unique
        # _, indices = np.unique(all_positions.round(UNIQUE_TOL), axis=0, return_index=True)
        # all_positions = np.array([all_positions[index] for index in sorted(indices)])
        # all_energies = np.array([all_energies[index] for index in sorted(indices)])
        all_positions = cart2sphA(all_positions)
        x = all_positions[:, 2]
        y = all_positions[:, 1]
        if self.energies is None:
            self.ax.scatter(*all_positions.T, c="black")
        else:
            cmap = ListedColormap((sns.color_palette("coolwarm", 256).as_hex()))
            im = self.ax.scatter(x, y, c=all_energies, cmap=cmap)
            cbar = self.fig.colorbar(im, ax=self.ax)
            cbar.set_label(f"{self.property} {self.unit}")
            self.ax.set_title(f"N = {self.selected_Ns[self.N_index]}")
        plt.grid(True)

    def create(self, *args, **kwargs):
        AbstractPlot.create(self, *args, projection="hammer", **kwargs)


class TrajectoryEnergyMultiPlot(AbstractMultiPlot):

    def __init__(self, list_plots: List[TrajectoryEnergyPlot], **kwargs):
        super().__init__(list_plots, **kwargs)

    def create(self, *args, **kwargs):
        super().create(*args, projection="3d", **kwargs)


class HammerProjectionMultiPlot(AbstractMultiPlot):

    def __init__(self, list_plots: List[HammerProjectionTrajectory], plot_type="hammer", figsize=None,
                 n_rows=1, n_columns=5, **kwargs):
        if figsize is None:
            figsize = (DIM_LANDSCAPE[0]*n_columns, DIM_LANDSCAPE[1]*n_rows)
        super().__init__(list_plots, plot_type=plot_type, n_rows=n_rows, n_columns=n_columns, figsize=figsize, **kwargs)

    def create(self, *args, projection="hammer", **kwargs):
        super().create(*args, projection=projection, **kwargs)


def create_hammer_multiplot(data_name, Ns=None):
    list_single_plots = []
    max_index = 5 if Ns is None else len(Ns)
    for i in range(max_index):
        tep = HammerProjectionTrajectory(data_name, selected_Ns=Ns)
        tep.N_index = i
        tep.add_energy_information(f"{PATH_INPUT_ENERGIES}{data_name}.xvg")
        list_single_plots.append(tep)
    HammerProjectionMultiPlot(list_single_plots, n_columns=max_index, n_rows=1).create_and_save()

class VoranoiConvergencePlot(AbstractPlot):

    def __init__(self, data_name: str, style_type=None, plot_type="areas"):
        super().__init__(data_name, dimensions=2, style_type=style_type, plot_type=plot_type)

    def _prepare_data(self) -> object:
        return pd.read_csv(f"{PATH_OUTPUT_CELLS}{self.data_name}.csv")

    def _plot_data(self, color=None, **kwargs):
        N_points = CELLS_DF_COLUMNS[0]
        voranoi_areas = CELLS_DF_COLUMNS[2]
        ideal_areas = CELLS_DF_COLUMNS[3]
        time = CELLS_DF_COLUMNS[4]
        voranoi_df = self._prepare_data()
        sns.lineplot(data=voranoi_df, x=N_points, y=voranoi_areas, errorbar="sd", color=color, ax=self.ax)
        sns.scatterplot(data=voranoi_df, x=N_points, y=voranoi_areas, alpha=0.8, color="black", ax=self.ax, s=1)
        sns.scatterplot(data=voranoi_df, x=N_points, y=ideal_areas, color="black", marker="x", ax=self.ax)
        ax2 = self.ax.twinx()
        ax2.set_yscale('log')
        ax2.set_ylim(10**-3, 10**3)
        sns.lineplot(data=voranoi_df, x=N_points, y=time, color="black", ax=ax2)

    def create(self, *args, **kwargs):
        super().create(*args, **kwargs)
        self.ax.set_xscale('log')
        #self.ax.set_ylim(0.01, 2)


class GridColoredWithAlphaPlot(GridPlot):
    def __init__(self, data_name, vector: np.ndarray, alpha_set: list, plot_type: str = "colorful_grid", **kwargs):
        super().__init__(data_name, plot_type=plot_type, **kwargs)
        self.alpha_central_vector = vector
        self.alpha_set = alpha_set
        self.alpha_set.sort()
        self.alpha_set.append(pi)

    def _plot_data(self, color=None, **kwargs):
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
            ratios_df = pd.read_csv(my_grid.statistics_path, dtype=float)
        except FileNotFoundError:
            my_grid.save_statistics()
            ratios_df = pd.read_csv(my_grid.statistics_path, dtype=float)
        return ratios_df

    def _plot_data(self, color=None, **kwargs):
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


class PolytopePlot(Plot3D):

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
        super().__init__(data_name, fig_path=PATH_OUTPUT_PLOTS, plot_type=plot_type, style_type=["empty"],
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


class EnergyConvergencePlot(AbstractPlot):

    def __init__(self, data_name: str, test_Ns=None, property_name="Potential", no_convergence=False,
                 plot_type="energy_convergence", **kwargs):
        self.test_Ns = test_Ns
        self.property_name = property_name
        self.unit = None
        self.no_convergence = no_convergence
        super().__init__(data_name, plot_type=plot_type, **kwargs)

    def _prepare_data(self) -> pd.DataFrame:
        file_name = f"{PATH_INPUT_ENERGIES}{self.data_name}.xvg"
        file_parsed = XVGParser(file_name)
        self.property_name, correct_column = file_parsed.get_column_index_by_name(self.property_name)
        self.unit = file_parsed.get_y_unit()
        df = pd.DataFrame(file_parsed.all_values[:, correct_column], columns=[self.property_name])
        # select points that fall within each entry in test_Ns
        self.test_Ns = test_or_create_Ns(len(df), self.test_Ns)
        if self.no_convergence:
            self.test_Ns = [self.test_Ns[-1]]
        points_up_to_Ns(df, self.test_Ns, target_column=self.property_name)
        return df

    def _plot_data(self, **kwargs):
        df = self._prepare_data()
        new_column_names = [f"{i}" for i in self.test_Ns]
        sns.violinplot(df[new_column_names], ax=self.ax, scale="area", inner="stick")
        self.ax.set_xlabel("N")
        if self.unit:
            self.ax.set_ylabel(f"{self.property_name} [{self.unit}]")
        else:
            self.ax.set_ylabel(self.property_name)


def points_up_to_Ns(df: pd.DataFrame, Ns: ArrayLike, target_column: str):
    """
    Introduce a new column into the dataframe for every N  in the Ns list. The new columns contain the value from
     target_column if the row index is <= N. Every value in the Ns list must be smaller or equal the length of the
     DataFrame. The DataFrame is changed in-place.

    Why is this helpful? For plotting convergence plots using more and more of the data.

    Example:

    # Initial dataframe df:
        index   val1    val2
        1        a       b
        2        c       d
        3        e       f
        4        g       h

    # After applying points_up_to_Ns(df, Ns=[1, 3, 4], target_column='val2'):
        index   val1    val2    1       3       4
        1        a       b      b       b       b
        2        c       d              d       d
        3        e       f              f       f
        4        g       h                      h

    # Plotting the result as separate items
    sns.violinplot(df["1", "3", "4"], ax=self.ax, scale="area", inner="stick")
    """
    # create helper columns under_N where the value is 1 if index <= N
    selected_Ns = np.zeros((len(Ns), len(df)))
    for i, point in enumerate(Ns):
        selected_Ns[i][:point] = 1
    column_names = [f"under {i}" for i in Ns]
    df[column_names] = selected_Ns.T
    # create the columns in which values from target_column are copied
    for point in Ns:
        df[f"{point}"] = df.where(df[f"under {point}"] == 1)[target_column]
    # remover the helper columns
    return df


def test_or_create_Ns(max_N: int, Ns: ArrayLike = None,  num_test_points=5) -> ArrayLike:
    if Ns is None:
        Ns = np.linspace(0, max_N, num_test_points+1, dtype=int)[1:]
    else:
        assert np.all([np.issubdtype(x, np.integer) for x in Ns]), "All tested Ns must be integers."
        assert np.max(Ns) <= max_N, "Test N cannot be larger than the number of points"
    return Ns


def groupby_min_body_energy(df: pd.DataFrame, target_column: str, N_b: int) -> pd.DataFrame:
    """
    Take a dataframe with positions and energies and return only one row per COM position of the second molecule,
    namely the one with lowest energy.

    Args:
        df: dataframe resulting from a Pseudotrajectory with typical looping over:
            rotations of origin
                rotations of body
                    translations must be already filtered out by this point!
        target_column: name of the column in which energy values are found
        N_b: number of rotations around body for this PT.

    Returns:
        a DataFrame with a number of rows equal original_num_of_rows // N_b
    """
    # in case that translations have been removed, the index needs to be re-set
    df.reset_index(inplace=True, drop=True)
    start_len = len(df)
    new_df = df.loc[df.groupby(df.index // N_b)[target_column].idxmin()]
    assert len(new_df) == start_len // N_b
    return new_df


def get_only_one_translation_distance(df: pd.DataFrame, N_t: int, distance_index=0) -> pd.DataFrame:
    start_len = len(df)
    assert distance_index < N_t, f"Only {N_t} different distances available, " \
                                 f"cannot get the distance with index {distance_index}"
    new_df = df.iloc[range(0, len(df), N_t)]
    assert len(new_df) == start_len // N_t
    return new_df


if __name__ == "__main__":
    # H2O_H2O_o_ico_50_b_ico_10_t_902891566
    # tep = TrajectoryEnergyPlot("H2O_H2O_o_ico_500_b_ico_5_t_3830884671")
    # tep.add_energy_information("input/H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xvg")
    # tep.create_and_save(animate_rot=True)
    #create_trajectory_energy_multiplot("H2O_H2O_o_ico_500_b_ico_5_t_3830884671", animate_rot=False)
    #AlphaConvergencePlot("systemE", style_type=["talk"]).create_and_save(equalize=True, title="Convergence of systemE")

    hpt = HammerProjectionTrajectory("H2O_H2O_run")
    hpt.add_energy_information("input/H2O_H2O_run.xvg")
    hpt.create_and_save()

    # hpt = HammerProjectionTrajectory("H2O_H2O_o_ico_500_b_ico_5_t_3830884671")
    # hpt.add_energy_information("input/H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xvg")
    # hpt.create_and_save()
    #
    # create_hammer_multiplot("H2O_H2O_o_ico_500_b_ico_5_t_3830884671")

    #EnergyConvergencePlot("full_energy_H2O_H2O", test_Ns=(5, 10, 20, 30, 40, 50), property_name="LJ (SR)").create_and_save()
    # EnergyConvergencePlot("full_energy_protein_CL", test_Ns=(10, 50, 100, 200, 300, 400, 500)).create_and_save()
    # EnergyConvergencePlot("full_energy2", test_Ns=(100, 500, 800, 1000, 1500, 2000, 3000, 3600)).create_and_save()

    # FullGrid(o_grid_name="cube4D_12", b_grid_name="zero", t_grid_name='range(1, 5, 2)')
    # PositionGridPlot("position_grid_o_cube4D_12_b_zero_1_t_3203903466", cell_lines=True).create_and_save(
    #     animate_rot=True, animate_seq=False)
    # PanelPlot([VoranoiConvergencePlot(f"{i}_1_10_100") for i in GRID_ALGORITHMS[:-1]]).create_and_save()