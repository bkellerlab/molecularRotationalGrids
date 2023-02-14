from abc import ABC
from typing import Union, List, Tuple, Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, Animation
from matplotlib.pyplot import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy._typing import NDArray

from molgri.constants import DIM_SQUARE, DEFAULT_DPI, EXTENSION_FIGURES, DEFAULT_DPI_MULTI
from molgri.paths import PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS


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
        path_plot = f"{self.fig_path}{self.data_name}_{plot_type_name}"
        self.fig.tight_layout()
        plt.savefig(f"{path_plot}.{save_ending}", dpi=dpi, bbox_inches='tight')
        plt.close()

    def _save_animation_type(self, animation: Animation, ani_type_name: str, fps: int = 10,
                             dpi: int = DEFAULT_DPI_MULTI):
        writergif = PillowWriter(fps=fps, bitrate=-1)
        # noinspection PyTypeChecker
        animation.save(f"{self.ani_path}{self.data_name}_{ani_type_name}.gif", writer=writergif, dpi=dpi)

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

    def _animate_figure_view(self, fig, ax) -> FuncAnimation:
        """
        Call after you have created some 3D figure.
        Rotate the 3D figure for 360 degrees around itself and save the animation.
        """

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
                           remove_midlabels: bool = True, projection: str = None):
        """This method should be run in all sub-modules to set-up the axes"""
        if plotting_kwargs is None:
            plotting_kwargs = dict()
        _set_style_and_context(context=self.list_plots[0].context, color_style=self.list_plots[0].color_style)
        self.__create_fig_ax(all_ax=all_ax, projection=projection)
        for ax, subplot in zip(self.all_ax.ravel(), self.list_plots):
            plot_func = getattr(subplot, plotting_method)
            plot_func(fig=self.fig, ax=ax, save=False, **plotting_kwargs)
        if remove_midlabels:
            self.__remove_midlabels()

    def __create_fig_ax(self, all_ax=None, sharex="all", sharey="all", projection=None):
        if all_ax:
            self.fig = plt.figure(figsize=self.figsize)
            self.all_ax = all_ax
        else:
            self.fig, self.all_ax = plt.subplots(self.n_rows, self.n_columns, subplot_kw={'projection': projection},
                                                 sharex=sharex, sharey=sharey, figsize=self.figsize)

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
            if isinstance(subaxis, Axes3D) and subaxis.z_lim[1] > z_lim[1]:
                z_lim[1] = subaxis.get_zlim()[1]
        # change them all
        for i, subaxis in enumerate(self.all_ax.ravel()):
            if x_ax:
                subaxis.set_xlim(*x_lim)
            if y_ax:
                subaxis.set_ylim(*y_lim)
            if isinstance(subaxis, Axes3D) and z_ax:
                subaxis.set_zlim(*z_lim)

    def animate_figure_view(self, plot_type: str) -> FuncAnimation:
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
        ani_path = self.list_plots[0].ani_path
        # noinspection PyTypeChecker
        anim.save(f"{ani_path}{self.data_name}_{plot_type}.gif", writer=writergif, dpi=400)
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




class PanelRepresentationCollection(MultiRepresentationCollection):

    def __init__(self, data_name, list_plots: List[RepresentationCollection], landscape=True, figsize=None):
        if landscape:
            n_rows = 2
            n_columns = 3
        else:
            n_rows = 3
            n_columns = 2
        super().__init__(data_name, list_plots, n_columns=n_columns, n_rows=n_rows, figsize=figsize)
