from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib import ticker
from mpl_toolkits.mplot3d.axes3d import Axes3D

from molgri.constants import DIM_SQUARE, DEFAULT_DPI, COLORS, EXTENSION_FIGURES, FULL_GRID_ALG_NAMES
from molgri.paths import PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS


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
                self.ax = self.fig.add_subplot(111, projection=projection)
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
        subplot_dims_equal = [list_plot.dimensions == self.dimensions for list_plot in list_plots]
        assert np.all(subplot_dims_equal), "Plots cannot have various numbers of directions"
        self.n_rows = n_rows
        self.n_columns = n_columns
        assert self.n_rows*self.n_columns == len(list_plots), f"Specify the number of rows and columns that " \
                                                              f"corresponds to the number of the elements! " \
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
        self.fig.savefig(f"{fig_path}multi_{standard_name}_{self.plot_type}.{save_ending}", dpi=dpi,
                         bbox_inches='tight', **kwargs)


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
