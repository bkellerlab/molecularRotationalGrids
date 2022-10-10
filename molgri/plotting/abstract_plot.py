from abc import ABC, abstractmethod

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from molgri.my_constants import *
from molgri.paths import PATH_OUTPUT_GRIDORDER_ANI
from molgri.parsers.name_parser import NameParser
from molgri.plotting.plotting_helper_functions import set_axes_equal


def set_up_style(style_type: list):
    """
    Before creating fig and ax, set up background color (eg. dark) and context (eg. talk).

    Args:
        style_type: list of plot descriptions eg. ['dark', 'talk', 'half_empty']
    """
    sns.reset_orig()
    plt.style.use('default')
    if "dark" in style_type:
        plt.style.use('dark_background')
    if "talk" in style_type:
        sns.set_context("talk")


class AbstractPlot(ABC):
    """
    Most general plotting class (for one axis per plot). Set all methods that could be useful more than one time here.
    All other plots, including multiplots, inherit from this class.
    """

    def __init__(self, data_name: str, dimensions: int = 3, style_type: list = None, fig_path: str = None,
                 ani_path: str = None, ax: plt.Axes = None, figsize: tuple = DIM_LANDSCAPE,
                 plot_type: str = "abs"):
        """
        Input all information that needs to be provided before fig and ax are created.

        Args:
            data_name: eg ico_500_full
            dimensions: 2 or 3
            style_type: a list of properties like 'dark', 'talk', 'empty' or 'half_empty'
            fig_path: folder to save figures if created
            ani_path: folder to save animations if created
            ax: enables to pass an already created axis - useful for PanelPlots
            figsize: forwarded to set up of the figure
            plot_type: str describing the plot function, added to the name of the plot
        """
        if style_type is None:
            style_type = ["white"]
        self.fig_path = fig_path
        self.ani_path = ani_path
        self.dimensions = dimensions
        self.style_type = style_type
        self.data_name = data_name
        self.parsed_data_name = NameParser(self.data_name)
        self.plot_type = plot_type
        self.figsize = figsize
        try:
            self.default_title = NAME2PRETTY_NAME[self.parsed_data_name.get_grid_type()]
        except ValueError:
            self.default_title = None
        # here change styles that need to be set before fig and ax are created
        set_up_style(style_type)
        # create the empty figure
        self.fig = None
        self.ax = ax

    def create(self, *args, equalize=False, neg_limit=None, pos_limit=None, x_label=None, y_label=None, z_label=None,
               title=None, save_fig=True, animate_rot=False, animate_seq=False, sci_limit_min=-4, sci_limit_max=4,
               save_ending="pdf", dpi=600, labelpad=0, pad_inches=0, sharex="all", sharey="all", close_fig=True,
               azim=-60, elev=30):
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
        if save_fig:
            self._save_plot(save_ending=save_ending, dpi=dpi, pad_inches=pad_inches)
        if close_fig:
            plt.close()
        if animate_rot:
            self.animate_figure_view()
        if animate_seq:
            self.animate_grid_sequence()

    def _create_fig_ax(self, sharex="all", sharey="all"):
        """
        The parameters need to stay there to be consistent with AbstractMultiPlot, but are not used.

        Args:
            sharex: if multiplots should share the same x axis
            sharey: if multiplots should share the same y axis
        """
        self.fig = plt.figure(figsize=self.figsize)
        if self.ax is None:
            if self.dimensions == 3:
                self.ax = self.fig.add_subplot(111, projection='3d')
            elif self.dimensions == 2:
                self.ax = self.fig.add_subplot(111)
            else:
                raise ValueError("Only 2 or 3 dimensions possible.")

    def _set_up_empty(self):
        """
        Second part of setting up the look of the plot, this time deleting unnecessary properties.
        If 'half_empty', remove ticks, if 'empty', also any shading of the background in 3D plots.
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
            self.ax.w_xaxis.set_pane_color(color)
            self.ax.w_yaxis.set_pane_color(color)
            self.ax.w_zaxis.set_pane_color(color)

    def _equalize_axes(self, neg_limit: float = None, pos_limit: float = None):
        """
        Makes x, y, (z) axis equally longs and if limits given, enforces them on all axes.

        Args:
            neg_limit: if set, this will be min x, y, (z) value of the plot
            pos_limit: if set, this will be max x, y, (z) value of the plot - if pos_limit set but neg_limit not,
                       neg_limit is set to -pos_limit
        """
        if self.dimensions == 3:
            set_axes_equal(self.ax)
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
        pass

    @abstractmethod
    def _plot_data(self, **kwargs):
        """Here, the plotting is implemented in subclasses."""
        pass

    def _create_labels(self, x_label=None, y_label=None, z_label=None, **kwargs):
        if x_label:
            self.ax.set_xlabel(x_label, **kwargs)
        if y_label:
            self.ax.set_ylabel(y_label, **kwargs)
        if z_label and self.dimensions == 3:
            self.ax.set_zlabel(z_label, **kwargs)

    def _create_title(self, title):
        if "talk" in self.style_type:
            self.ax.set_title(title, fontsize=15)
        else:
            self.ax.set_title(title)

    def _sci_ticks(self, neg_lim: int = -4, pos_lim: int = 4):
        try:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(neg_lim, pos_lim))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(neg_lim, pos_lim))
        except AttributeError:
            pass

    def animate_figure_view(self):
        """
        Rotate the 3D figure and save the animation.
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

    def animate_grid_sequence(self):
        """
        Animate how a grid is constructed - how each individual point is added.
        """
        self.ax = None
        self._create_fig_ax()
        self._set_up_empty()
        grid = self._prepare_data()

        def update(i):
            grayscale_color = 0.8 * (grid_plot[0, i] + 1) / 2 + 0.2 * (grid_plot[1, i] + 1) / 2
            new_point = self.ax.scatter(*grid_plot[:, i], color=str(grayscale_color),
                                   alpha=(1 - grayscale_color + 0.5) / 1.5)
            return new_point,

        # noinspection PyUnresolvedReferences
        if len(grid.shape) == 2:
            grid_plot = grid.T
        else:
            grid_plot = grid

        self.ax.view_init(elev=30, azim=30)
        self._equalize_axes(pos_limit=1, neg_limit=-1)
        #for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
        #    axis.set_major_locator(ticker.MaxNLocator(integer=True))
        ani = FuncAnimation(self.fig, func=update, frames=grid_plot.shape[1], interval=50, repeat=False)
        writergif = PillowWriter(fps=1, bitrate=-1)
        # noinspection PyTypeChecker
        ani.save(f"{PATH_OUTPUT_GRIDORDER_ANI}{self.data_name}_{self.plot_type}.gif", writer=writergif, dpi=400)
        plt.close()

    def _save_plot(self, save_ending="pdf", dpi=DEFAULT_DPI, **kwargs):
        self.fig.tight_layout()
        if self.fig_path:
            standard_name = self.parsed_data_name.get_standard_name()
            plt.savefig(f"{self.fig_path}{standard_name}_{self.plot_type}.{save_ending}", dpi=dpi, bbox_inches='tight',
                        **kwargs)
        else:
            raise ValueError("path to save the figure has not been selected!")
        plt.close()


