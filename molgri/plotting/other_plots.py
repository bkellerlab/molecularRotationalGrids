import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from matplotlib.pyplot import Figure, Axes
import matplotlib.colors as colors

from molgri.plotting.abstract import RepresentationCollection


class ArrayPlot(RepresentationCollection):

    """
    A tool for plotting arrays, eg by highlighting high and low values
    """

    def __init__(self, my_array: NDArray, *args, data_name="array", **kwargs):
        self.array = my_array
        super().__init__(data_name, *args, **kwargs)

    def make_heatmap_plot(self, ax: Axes = None, fig: Figure = None, save: bool = True):
        """
        This method draws the array and colors the fields according to their values (red = very large,
        blue = very small). Zero values are always white, negative ones always blue, positive ones always red.
        """
        self._create_fig_ax(fig=fig, ax=ax)
        if np.all(self.array < 0):
            cmap = "Blues"
            norm = None
        elif np.all(self.array > 0):
            cmap = "Reds"
            norm = None
        else:
            cmap = "bwr"
            norm = colors.TwoSlopeNorm(vcenter=0)
        sns.heatmap(self.array, cmap=cmap, ax=self.ax, xticklabels=False, yticklabels=False, norm=norm)

        self._equalize_axes()
        if save:
            self._save_plot_type("heatmap")
