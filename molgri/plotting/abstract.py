from abc import ABC
from typing import Union, List, Tuple, Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, Animation
from matplotlib.axes import Axes
from matplotlib import ticker
from mpl_toolkits.mplot3d.axes3d import Axes3D

from molgri.constants import DIM_SQUARE, DEFAULT_DPI, COLORS, EXTENSION_FIGURES, FULL_GRID_ALG_NAMES, DEFAULT_DPI_MULTI
from molgri.paths import PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS


class Representation:
    """
    This class represents a single axis instance. Individual methods in RepresentationCollection create a
    Representation.
    """

    def __init__(self, dimension, fig: plt.Figure = None, ax: Union[Axes, Axes3D] = None, projection: str = None,
                 figsize: tuple = None, context: str = None, color_style: str = None,
                 complexity_level: str = None):



        self.dimension = dimension
        self._set_style_and_context(context=context, color_style=color_style)
        self.fig, self.ax = self._create_fig_ax(fig=fig, ax=ax, projection=projection, figsize=figsize)
        self._set_complexity_level(complexity_level)
        self._set_style_part_2(color_style=color_style)



    def set_axis_limits(self, limits: tuple = None):
        """
        List limits in a 4- or 6-element tuple in the order: x_min, x_max, y_min, y_max, z_min, z_max.
        If a plot is 2D, z values should be omitted. If any value should not be set, use None for that element.

        Args:
            limits: a tuple including all 4 or 6 limits
        """
        assert len(limits) == 2*self.dimension, "Wrong number of limit values"

        x_min_limit = limits[0]
        x_max_limit = limits[1]
        y_min_limit = limits[2]
        y_max_limit = limits[3]

        self.ax.set_xlim(x_min_limit, x_max_limit)
        self.ax.set_ylim(y_min_limit, y_max_limit)

        if self.dimension == 3:
            z_min_limit = limits[4]
            z_max_limit = limits[5]
            self.ax.set_zlim(z_min_limit, z_max_limit)

    def set_labels(self, x_label: str = None, y_label: str = None, z_label: str = None, **kwargs):
        if x_label:
            self.ax.set_xlabel(x_label, **kwargs)
        if y_label:
            self.ax.set_ylabel(y_label, **kwargs)
        if z_label and self.dimension == 3:
            self.ax.set_zlabel(z_label, **kwargs)

    def set_title(self, title: str, **kwargs):
        self.ax.set_title(title, **kwargs)

    def equalize_axes(self):
        """
        Makes x, y, (z) axis equally longs and if limits given, enforces them on all axes.
        If you also use set_axis_limits, this should be run afterwards!
        """
        if self.dimension == 2:
            self.ax.set_aspect('equal')
        else:
            # because ax.set_aspect('equal') does not work for 3D axes
            self.ax.set_box_aspect(aspect=[1, 1, 1])
            x_lim, y_lim, z_lim = self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()
            all_ranges = abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]), abs(z_lim[1] - z_lim[0])
            x_middle, y_middle, z_middle = np.mean(x_lim), np.mean(y_lim), np.mean(z_lim)
            plot_range = 0.5 * max(all_ranges)
            self.ax.set_xlim3d([x_middle - plot_range, x_middle + plot_range])
            self.ax.set_ylim3d([y_middle - plot_range, y_middle + plot_range])
            self.ax.set_zlim3d([z_middle - plot_range, z_middle + plot_range])

    def save_plot(self, path: str = None, save_ending: str = EXTENSION_FIGURES, dpi: int = DEFAULT_DPI, **kwargs):
        self.fig.tight_layout()
        plt.savefig(f"{path}.{save_ending}", dpi=dpi, bbox_inches='tight', **kwargs)
        plt.close()

    def get_fig_ax(self) -> Tuple[plt.Figure, Union[Axes, Axes3D]]:
        return self.fig, self.ax


class RepresentationCollection(ABC):
    """
    Most general plotting class. Set all methods that could be useful for many various types of plots here.
    All other plots, including multiplots, inherit from this class.
    """

    def __init__(self, data_name: str, fig_path: str = PATH_OUTPUT_PLOTS,
                 ani_path: str = PATH_OUTPUT_ANIS, default_figsize: tuple = DIM_SQUARE,
                 default_context: str = "notebook", default_color_style: str = "white",
                 default_complexity_level="full", default_axes_limits: tuple = (None,)*6):
        """
        In the __init__, defaults should be set that will generally be used for an entire class of

        Args:
            data_name: keyword that combines all figures of a specific sub-class
            fig_path: folder to save figures if created, should be set in subclasses
            ani_path: folder to save animations if created, should be set in subclasses
            default_context: forwarded to sns.set_context -> should be one of: paper, notebook, talk, poster
            default_color_style: describes the general color scheme of the plots (dark, white)
            default_complexity_level: select from full, half_empty, empty

        """
        # defaults:
        if context is None:
            context = "notebook"
        if color_style is None:
            color_style = "white"
        if complexity_level is None:
            complexity_level = "full"

        self.data_name = data_name
        self.fig_path = fig_path
        self.ani_path = ani_path
        self.figsize = default_figsize
        self.context = default_context
        self.color_style = default_color_style
        self.complexity_level = default_complexity_level
        self.axes_limits = default_axes_limits



    # def _main_ticks(self, ax):
    #     for axis in [ax.xaxis, ax.yaxis]:
    #         axis.set_major_locator(ticker.MaxNLocator(integer=True))


    # def _set_up_empty(self):
    #     """
    #     Second part of setting up the look of the plot, this time deleting unnecessary properties.
    #     Keywords to change the look of the plot are taken from the property self.style_type.
    #     If 'half_empty', remove ticks, if 'empty', also any shading of the background in 3D plots.
    #     The option 'half_dark' changes background to gray, 'dark' to black.
    #     """
    #     if "empty" in self.style_type or "half_empty" in self.style_type:
    #         self.ax.set_xticks([])
    #         self.ax.set_yticks([])
    #
    #         if "empty" in self.style_type:
    #             self.ax.axis('off')
    #     color = (0.5, 0.5, 0.5, 0.7)
    #     if "half_dark" in self.style_type:
    #         self.ax.xaxis.set_pane_color(color)
    #         self.ax.yaxis.set_pane_color(color)
    #     return color



    # @abstractmethod
    # def _prepare_data(self) -> object:
    #     """
    #     This function should only be used by the self._plot_data method to obtain the data that we wanna plot.
    #
    #     Returns:
    #         dataframe, grid or similar construction
    #     """
    #
    # @abstractmethod
    # def _plot_data(self, **kwargs):
    #     """Here, the plotting is implemented in subclasses."""



    # @staticmethod
    # def _sci_ticks(neg_lim: int = -4, pos_lim: int = 4):
    #     try:
    #         plt.ticklabel_format(style='sci', axis='x', scilimits=(neg_lim, pos_lim))
    #         plt.ticklabel_format(style='sci', axis='y', scilimits=(neg_lim, pos_lim))
    #     except AttributeError:
    #         pass

    # def save_plot(self, name_addition: str = None, save_ending: str = EXTENSION_FIGURES, dpi: int = DEFAULT_DPI, **kwargs):
    #     self.fig.tight_layout()
    #     standard_name = self.data_name
    #     if name_addition is None:
    #         addition = ""
    #     else:
    #         addition = f"_{name_addition}"
    #     plt.savefig(f"{self.fig_path}{standard_name}{addition}.{save_ending}", dpi=dpi, bbox_inches='tight',
    #                 **kwargs)
    #     plt.close()


    def get_possible_title(self):
        return self.data_name

    def _create_fig_ax(self, dim: int, fig: plt.Figure = None, ax: Union[Axes, Axes3D] = None, projection=None,
                       figsize=None, complexity=None, color_style=None, context=None):

        if context is None:
            context = self.context
        if complexity is None:
            complexity = self.complexity_level
        if color_style is None:
            color_style = self.color_style
        if figsize is None:
            figsize = self.figsize

        self._set_style_and_context(context=context, color_style=color_style)
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            if dim == 2:
                ax = fig.add_subplot(111)
            elif dim == 3:
                ax = fig.add_subplot(111, projection=projection)
            else:
                raise ValueError(f"Dimensions must be in (2, 3), unknown value {dim}!")
        self._set_complexity_level(complexity)
        self._set_style_part_2(color_style=color_style)
        return fig, ax

    def _set_style_and_context(self, *, context: str = None, color_style: str = None):
        """
        This method should be run before an axis is created. Set up size of elements (paper, notebook, talk, poster),
        color scheme (white, dark ...) and similar. If nothing selected, use class defaults.

        Args:
            context: forwarded to sns.set_context -> should be one of: paper, notebook, talk, poster
            color_style: describes the general color scheme of the plots (dark, white)

        """
        sns.reset_orig()
        sns.set_context(context)
        plt.style.use('default')
        if "dark" in color_style:
            plt.style.use('dark_background')

    def _set_style_part_2(self, ax, dim, color_style):
        color = (0.5, 0.5, 0.5, 0.7)
        if color_style == "half_dark":
            ax.xaxis.set_pane_color(color)
            ax.yaxis.set_pane_color(color)
            if dim == 3:
                ax.zaxis.set_pane_color(color)

    def _set_complexity_level(self, complexity_level):
        if complexity_level == "empty" or complexity_level == "half_empty":
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            if self.dimension == 3:
                self.ax.set_zticks([])

            if complexity_level == "empty":
                self.ax.axis('off')

    def _save_plot_type(self, representation: Representation, plot_type_name: str, **kwargs):
        """To be run within specific methods that create plots"""
        path_plot = f"{self.fig_path}{self.data_name}_{plot_type_name}"
        representation.save_plot(path_plot, **kwargs)

    def _save_animation_type(self, animation: Animation, ani_type_name: str, fps: int = 10,
                             dpi: int = DEFAULT_DPI_MULTI):
        writergif = PillowWriter(fps=fps, bitrate=-1)
        # noinspection PyTypeChecker
        animation.save(f"{self.ani_path}{self.data_name}_{ani_type_name}.gif", writer=writergif, dpi=dpi)

    def animate_figure_view(self, representation: Representation) -> FuncAnimation:
        """
        Rotate the 3D figure for 360 degrees around itself and save the animation.
        """

        fig, ax = representation.get_fig_ax()

        def animate(frame):
            # rotate the view left-right
            ax.view_init(azim=2*frame)
            return fig

        anim = FuncAnimation(fig, animate, frames=180, interval=50)
        self._save_animation_type(anim, "rotated", fps=10, dpi=400)
        return anim


class MultiRepresentationCollection(ABC):

    def __init__(self, data_name, list_plots: List[RepresentationCollection], figsize=None,
                 n_rows: int = 4, n_columns: int = 2):
        self.n_rows = n_rows
        self.data_name = data_name
        self.n_columns = n_columns
        assert self.n_rows*self.n_columns == len(list_plots), f"Specify the number of rows and columns that " \
                                                              f"corresponds to the number of the elements! " \
                                                              f"{self.n_rows}x{self.n_columns}=/={len(list_plots)}"
        self.figsize = figsize
        if self.figsize is None:
            self.figsize = self.n_columns * DIM_SQUARE[0], self.n_rows * DIM_SQUARE[1]

        self.list_plots = list_plots

    def make_plot_for_all(self, dimensions, plotting_method: str, all_ax=None, plotting_kwargs=None,
                          remove_midlabels=True, titles=True):
        if plotting_kwargs is None:
            plotting_kwargs = dict()
        fig, all_ax = self._create_fig_ax(dimensions=dimensions, all_ax=all_ax)
        representations = []
        for i, subplot in enumerate(self.list_plots):
            plot_func = getattr(subplot, plotting_method)
            rep = plot_func(fig=fig, ax=all_ax.ravel()[i], save=False, **plotting_kwargs)
            if titles:
                rep.ax.set_title(subplot.get_possible_title())
            representations.append(rep)
        if remove_midlabels:
            self._remove_midlabels(all_ax, dimensions)
        return fig, all_ax, representations

    def _create_fig_ax(self, dimensions, all_ax=None, sharex="all", sharey="all"):
        if all_ax:
            fig = plt.figure(figsize=self.figsize)
        else:
            if dimensions == 3:
                # for 3D plots it makes no sense to share axes
                fig, all_ax = plt.subplots(self.n_rows, self.n_columns, subplot_kw={'projection': "3d"},
                                                     figsize=self.figsize)
            elif dimensions == 2:
                fig, all_ax = plt.subplots(self.n_rows, self.n_columns, sharex=sharex, sharey=sharey,
                                          figsize=self.figsize)
            else:
                raise ValueError("Only 2 or 3 dimensions possible.")
        return fig, all_ax

    def _remove_midlabels(self, all_ax, dimensions):
        if dimensions == 2:
            if self.n_rows > 1 and self.n_columns > 1:
                for my_a in all_ax[:, 1:]:
                    for subax in my_a:
                        subax.set_ylabel("")
                for my_a in all_ax[:-1, :]:
                    for subax in my_a:
                        subax.set_xlabel("")
            else:
                for my_a in all_ax[1:]:
                    my_a.set_ylabel("")

    def unify_axis_limits(self, x_ax=False, y_ax=False, z_ax=False):
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


    def create_and_save(self, *args, close_fig=True, **kwargs):
        self.create(*args, **kwargs)
        self.save_multiplot()
        if close_fig:
            plt.close()

    def _save_multiplot(self, fig, multiplot_type, save_ending: str = EXTENSION_FIGURES, dpi: int = DEFAULT_DPI, **kwargs):
        fig.tight_layout()
        fig_path = self.list_plots[0].fig_path
        standard_name = self.list_plots[-1].data_name
        fig.savefig(f"{fig_path}{self.data_name}_{multiplot_type}.{save_ending}", dpi=dpi,
                         bbox_inches='tight', **kwargs)


class PanelRepresentationCollection(MultiRepresentationCollection):

    def __init__(self, data_name, list_plots: List[RepresentationCollection], landscape=True, figsize=None):
        if landscape:
            n_rows = 2
            n_columns = 3
        else:
            n_rows = 3
            n_columns = 2
        super().__init__(data_name, list_plots, n_columns=n_columns, n_rows=n_rows, figsize=figsize)
