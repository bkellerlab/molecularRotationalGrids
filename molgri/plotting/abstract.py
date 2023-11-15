"""
Abstract implementation of plots and multi-plots.

All other plots extend RepresentationCollection. Each RepresentationCollection has a set of methods used to view the
data of the object with which the Collection is initiated in different ways. Plotting methods always posses ax and fig
arguments so that several plots can be plotted on top of each other or next to each other.

MultiRepresentationCollection provides functionality for combining several plots in one image.
PanelRepresentationCollection specifically creates one sub-plot for each grid-generating algorithm.
"""

from abc import ABC
from typing import Union, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, Animation
from matplotlib.pyplot import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray
from IPython import display
from scipy.spatial import geometric_slerp

from molgri.constants import DIM_SQUARE, DEFAULT_DPI, EXTENSION_FIGURES, DEFAULT_DPI_MULTI, EXTENSION_ANIMATIONS
from molgri.logfiles import paths_free_4_all
from molgri.paths import PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS
from molgri.space.utils import normalise_vectors


def _set_style_and_context(context: str = None, color_style: str = None):
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


class RepresentationCollection(ABC):
    """
    Most general plotting class. Set all methods that could be useful for many various types of plots here.
    All other plots, including multiplots, inherit from this class.
    """

    def __init__(self, data_name: str, fig_path: str = PATH_OUTPUT_PLOTS,
                 ani_path: str = PATH_OUTPUT_ANIS, default_figsize: tuple = DIM_SQUARE,
                 default_context: str = None, default_color_style: str = None,
                 default_complexity_level: str = None, default_axes_limits: tuple = (None,)*6):
        """
        In the __init__, defaults should be set that will generally be used for an entire class of figures - but it
        is still possible to not use defaults for some of the plots.

        Args:
            data_name: keyword that combines all figures of a specific sub-class
            fig_path: folder to save figures if created, should be set in subclasses
            ani_path: folder to save animations if created, should be set in subclasses
            default_context: forwarded to sns.set_context -> should be one of: paper, notebook, talk, poster
            default_color_style: describes the general color scheme of the plots (dark, white)
            default_complexity_level: select from full, half_empty, empty

        """
        # defaults
        if default_context is None:
            default_context = "notebook"
        if default_color_style is None:
            default_color_style = "white"
        if default_complexity_level is None:
            default_complexity_level = "full"

        self.data_name = data_name
        self.fig_path = fig_path
        self.ani_path = ani_path
        self.figsize = default_figsize
        self.context = default_context
        self.color_style = default_color_style
        self.complexity_level = default_complexity_level
        self.axes_limits = default_axes_limits

        # current values, will be overwritten by each new plot
        self.ax = None
        self.fig = None

    ##################################################################################################################
    #                                   SET-UP AXES & SAVE IN THE END
    ##################################################################################################################

    def _create_fig_ax(self, fig: Figure = None, ax: Union[Axes, Axes3D] = None, **creation_kwargs):
        """
        This function sets up a plot will create fig and ax in appropriate style. Each plotting function must call it.
        """
        context = creation_kwargs.pop("context", self.context)
        complexity = creation_kwargs.pop("complexity", self.complexity_level)
        color_style = creation_kwargs.pop("color_style", self.color_style)
        figsize = creation_kwargs.pop("figsize", self.figsize)
        projection = creation_kwargs.pop("projection", None)

        _set_style_and_context(context=context, color_style=color_style)
        if fig is None:
            self.fig = plt.figure(figsize=figsize)
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.add_subplot(111, projection=projection)
        else:
            self.ax = ax
        self.__set_complexity_level(complexity)
        self.__set_style_part_2(color_style=color_style)

    def _save_plot_type(self, plot_type_name: str, **saving_kwargs):
        save_ending = saving_kwargs.pop("save_ending", EXTENSION_FIGURES)
        dpi = saving_kwargs.pop("dpi", DEFAULT_DPI)
        path = paths_free_4_all([f"{self.data_name}_{plot_type_name}"], [save_ending], [self.fig_path])[0]
        self.fig.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()

    def _save_animation_type(self, animation: Animation, ani_type_name: str, fps: int = 10,
                             dpi: int = DEFAULT_DPI_MULTI):
        path = paths_free_4_all([f"{self.data_name}_{ani_type_name}"], [EXTENSION_ANIMATIONS], [self.ani_path])[0]
        writergif = PillowWriter(fps=fps, bitrate=-1)
        # noinspection PyTypeChecker
        animation.save(path, writer=writergif, dpi=dpi)

    def __set_complexity_level(self, complexity_level: str):
        if complexity_level == "empty" or complexity_level == "half_empty":
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            if isinstance(self.ax, Axes3D):
                self.ax.set_zticks([])

            if complexity_level == "empty":
                self.ax.axis('off')

    def __set_style_part_2(self, color_style: str):
        color = (0.5, 0.5, 0.5, 0.7)
        if color_style == "half_dark":
            self.ax.xaxis.set_pane_color(color)
            self.ax.yaxis.set_pane_color(color)
            if isinstance(self.ax, Axes3D):
                self.ax.zaxis.set_pane_color(color)

    ##################################################################################################################
    #                             USEFUL FUNCTIONS TO BE CALLED BY SUB-CLASSES
    ##################################################################################################################

    def get_possible_title(self):
        return self.data_name

    def _equalize_axes(self):
        """
        Makes x, y, (z) axis equally longs and if limits given, enforces them on all axes.
        If you also use set_axis_limits, this should be run afterwards!
        """
        if isinstance(self.ax, Axes3D):
            # because ax.set_aspect('equal') does not work for 3D axes
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

    def _set_axis_limits(self, limits: tuple = None):
        """
        List limits in a 4- or 6-element tuple in the order: x_min, x_max, y_min, y_max, z_min, z_max.
        If a plot is 2D, z values will be ignored. If any value should not be set, use None for that element.

        Args:
            limits: a tuple including all 4 or 6 limits
        """
        if limits is None:
            limits = self.axes_limits
        assert len(limits) == 4 or len(limits) == 6, "Wrong number of limit values"

        x_min_limit = limits[0]
        x_max_limit = limits[1]
        y_min_limit = limits[2]
        y_max_limit = limits[3]

        self.ax.set_xlim(x_min_limit, x_max_limit)
        self.ax.set_ylim(y_min_limit, y_max_limit)

        if isinstance(self.ax, Axes3D):
            z_min_limit = limits[4]
            z_max_limit = limits[5]
            self.ax.set_zlim(z_min_limit, z_max_limit)

    def _animate_figure_view(self, fig, ax, ani_name="rotated", **kwargs) -> FuncAnimation:
        """
        Call after you have created some 3D figure.
        Rotate the 3D figure for 360 degrees around itself and save the animation.
        """

        def animate(frame):
            # rotate the view left-right
            ax.view_init(azim=2*frame)
            return fig

        anim = FuncAnimation(fig, animate, frames=180, interval=50)
        dpi = kwargs.pop("dpi", 200)
        self._save_animation_type(anim, ani_name, fps=10, dpi=dpi)
        return anim


class MultiRepresentationCollection(ABC):

    def __init__(self, data_name, list_plots: List[RepresentationCollection], figsize=None,
                 n_rows: int = 4, n_columns: int = 2):
        self.n_rows = n_rows
        self.data_name = data_name
        self.n_columns = n_columns
        assert self.n_rows*self.n_columns == len(list_plots), f"{self.n_rows}x{self.n_columns}=/={len(list_plots)}"
        self.figsize = figsize
        if self.figsize is None:
            self.figsize = self.n_columns * DIM_SQUARE[0], self.n_rows * DIM_SQUARE[1]

        self.list_plots = list_plots

        # current values, to be overwritten with each plotting function
        self.fig = None
        self.all_ax = None

    ##################################################################################################################
    #                                   SET-UP AXES & SAVE IN THE END
    ##################################################################################################################

    def _make_plot_for_all(self, plotting_method: str, all_ax: NDArray = None, plotting_kwargs: dict = None,
                           remove_midlabels: bool = True, projection: str = None, creation_kwargs: dict = None,
                           fig: Figure = None):
        """This method should be run in all sub-modules to set-up the axes"""
        if plotting_kwargs is None:
            plotting_kwargs = dict()
        if creation_kwargs is None:
            creation_kwargs = dict()
        _set_style_and_context(context=self.list_plots[0].context, color_style=self.list_plots[0].color_style)
        if fig is None or all_ax is None:
            self.__create_fig_ax(all_ax=all_ax, projection=projection, **creation_kwargs)
        else:
            self.fig=fig
            self.all_ax = all_ax
        for ax, subplot in zip(self.all_ax.ravel(), self.list_plots):
            plot_func = getattr(subplot, plotting_method)
            plot_func(fig=self.fig, ax=ax, save=False, **plotting_kwargs)
        if remove_midlabels:
            self.__remove_midlabels()

    def __create_fig_ax(self, all_ax=None, sharex="all", sharey="all", projection=None, figsize=None):
        if figsize is None:
            figsize = self.figsize
        if all_ax:
            self.fig = plt.figure(figsize=figsize)
            self.all_ax = all_ax
        else:
            self.fig, self.all_ax = plt.subplots(self.n_rows, self.n_columns, subplot_kw={'projection': projection},
                                                 sharex=sharex, sharey=sharey, figsize=figsize)

    def __remove_midlabels(self):
        if not isinstance(self.all_ax[0], Axes3D):
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

    def _save_multiplot(self, multiplot_type, save_ending: str = EXTENSION_FIGURES, dpi: int = DEFAULT_DPI, **kwargs):
        self.fig.tight_layout()
        fig_path = self.list_plots[0].fig_path
        self.fig.savefig(f"{fig_path}{self.data_name}_{multiplot_type}.{save_ending}", dpi=dpi,
                         bbox_inches='tight', **kwargs)
        plt.close()

    def save_multianimation(self, animation, plot_type, dpi=100, fps=10):
        writergif = PillowWriter(fps=fps, bitrate=-1)
        ani_path = self.list_plots[0].ani_path
        # noinspection PyTypeChecker
        animation.save(f"{ani_path}{self.data_name}_{plot_type}.gif", writer=writergif, dpi=dpi)

    ##################################################################################################################
    #                             USEFUL FUNCTIONS TO BE CALLED BY SUB-CLASSES
    ##################################################################################################################

    def add_titles(self, list_titles, **title_kwargs):
        for title, ax in zip(list_titles, self.all_ax.ravel()):
            ax.set_title(title, **title_kwargs)

    def set_log_scale(self, x_axis=True, y_axis=True):
        for ax in self.all_ax.ravel():
            if x_axis:
                ax.set_xscale("log")
            if y_axis:
                ax.set_yscale("log")

    def add_legend(self, **legend_kwargs):
        """
        This adds one overall legend to the entire multiplot, only using unique handles.
        """
        all_handles = []
        all_labels = []
        for ax in self.all_ax.ravel():
            handles, labels = ax.get_legend_handles_labels()
            all_labels.extend(labels)
            all_handles.extend(handles)
        # remove duplicates
        unique = [(h, l) for i, (h, l) in enumerate(zip(all_handles, all_labels)) if l not in all_labels[:i]]
        self.fig.legend(*zip(*unique), **legend_kwargs)

    def unify_axis_limits(self, x_ax=True, y_ax=True, z_ax=True):
        """
        Unifies the min and max ranges of all subplots for selected axes.
        """
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
            if isinstance(subaxis, Axes3D) and subaxis.get_zlim()[0] < z_lim[0]:
                z_lim[0] = subaxis.get_zlim()[0]
            if isinstance(subaxis, Axes3D) and subaxis.get_zlim()[1] > z_lim[1]:
                z_lim[1] = subaxis.get_zlim()[1]
        # change them all
        for i, subaxis in enumerate(self.all_ax.ravel()):
            if x_ax:
                subaxis.set_xlim(*x_lim)
            if y_ax:
                subaxis.set_ylim(*y_lim)
            if isinstance(subaxis, Axes3D) and z_ax:
                subaxis.set_zlim(*z_lim)

    def animate_figure_view(self, plot_type: str, dpi=100) -> FuncAnimation:
        """
        Rotate all 3D figures for 360 degrees around themselves and save the animation.
        """

        def animate(frame):
            # rotate the view left-right
            for ax in self.all_ax.ravel():
                ax.view_init(azim=2*frame)
            return self.fig

        anim = FuncAnimation(self.fig, animate, frames=180, interval=50)
        self.save_multianimation(anim, plot_type=plot_type, dpi=dpi)
        return anim

    def add_colorbar(self, **cbar_kwargs):
        orientation = cbar_kwargs.pop("orientation", "vertical")
        cbar_label = cbar_kwargs.pop("cbar_label", None)
        all_collections = []
        for ax in self.all_ax.ravel():
            # this doesn't overwrite, but adds points to the collection!
            all_collections = ax.collections[0]

        cbar = self.fig.colorbar(all_collections, ax=self.all_ax[-1])
        # if orientation == "horizontal":
        #     pos = self.all_ax.ravel()[-1].get_position()
        #     cax = self.fig.add_axes([pos.x0 -0.05, pos.y0 -0.1, 0.3, 0.03])
        #
        # else:
        #     pos = self.all_ax.ravel()[-1].get_position()
        #     cax = self.fig.add_axes([pos.x1 - 1, pos.y0 + 0.0, 0.03, 0.4])
        # cbar = self.fig.colorbar(all_collections, cax=cax, orientation=orientation)
        # if orientation == "vertical":
        #     cax.yaxis.set_label_position('left')
        # if cbar_label:
        #     cbar.set_label(cbar_label)


class PanelRepresentationCollection(MultiRepresentationCollection):

    def __init__(self, data_name, list_plots: List[RepresentationCollection], landscape=True, figsize=None):
        if landscape:
            n_rows = 2
            n_columns = 3
        else:
            n_rows = 3
            n_columns = 2
        super().__init__(data_name, list_plots, n_columns=n_columns, n_rows=n_rows, figsize=figsize)


def show_anim_in_jupyter(anim):
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    # necessary to not create an extra empty window
    plt.close()


def plot_voronoi_cells(sv, ax, plot_vertex_points=True, colors=None):
    sv.sort_vertices_of_regions()
    t_vals = np.linspace(0, 1, 2000)
    # plot Voronoi vertices
    if plot_vertex_points:
        ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')
    # indicate Voronoi regions (as Euclidean polygons)
    for i, region in enumerate(sv.regions):
        n = len(region)
        for j in range(n):
            start = sv.vertices[region][j]
            end = sv.vertices[region][(j + 1) % n]
            norm = np.linalg.norm(start)
            result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)
            ax.plot(norm * result[..., 0], norm * result[..., 1], norm * result[..., 2], c='k')
        if colors:

            associated_vertices = sv.vertices[region]
            within_region = np.vstack([sv.points[i], associated_vertices])
            #if np.any(additional_assignments[i]):
            #    within_region = np.vstack([additional_assignments[i], within_region])
            #my_convex_hull = ConvexHull(within_region, qhull_options='QJ')

            polygon = Poly3DCollection(sv.vertices[region], alpha=0.5)
            polygon.set_color(colors[i])
            ax.add_collection3d(polygon)
