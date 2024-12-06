from typing import Optional

import MDAnalysis as mda
import matplotlib
import nglview as nv
import ipywidgets as widgets
import numpy as np
from IPython.core.display import display
from numpy.typing import ArrayLike, NDArray


class ViewManager:
    """
    NGLViewer is very useful but not very convinient for displaying particular frames of a trajectory together,
    in particular color schemes etc. This class accepts a MDA Universe and knows how to extract frames from it.
    The plotting functions then accept indices (one or several) for this trajectory and display them in a
    particular way (overlapping, sequential ...)
    """

    def __init__(self, u: mda.Universe):
        self.u = u
        self.fresh_view()


    def fresh_view(self):
        """
        Run this when you want to start a new view and discard an old one.
        """
        self.current_magnitudes: Optional[NDArray] = None
        self.current_opacities: Optional[NDArray] = None
        self.view = nv.NGLWidget()

    def get_ith_frame(self, i: int) -> mda.Universe:
        """
        The most important method acting on self.u. Get the Universe object containing of all the atoms
        but only a single frame (i-th frame).

        Args:
            i (int): the index of the frame wanted

        Returns:
            a Universe containing the i-th frame of self.u
        """
        self.u.trajectory[i]
        new_u = mda.Merge(self.u.atoms)
        return new_u

    def add_vector(self, vector: NDArray, label = None, rgb=(0,0,0)):
        self.view.shape.add_arrow([0, 0, 0], vector, rgb, 0.1)

        if label:
            self.view.shape.add_label(vector/2, rgb, 1.5, label)

    def add_principal_axes(self, pa_center, pa_array):
        """
        Helper method. Add the x, y and z axes at origin to a NGLView.
        """

        # arguments of add_arrow are: start position, end position, color (in RGB), radius of arrow head
        # arguments of add_label are: position, color (in RGB), size, text

        # X-axis is red
        self.view.shape.add_arrow(pa_center, pa_center+pa_array[0], [1, 0, 0], 0.1)
        self.view.shape.add_label(pa_center+pa_array[0], [1, 0, 0], 1.5, 'q1')

        # Y-axis is green
        self.view.shape.add_arrow(pa_center, pa_center+pa_array[1], [0, 1, 0], 0.1)
        self.view.shape.add_label(pa_center+pa_array[1], [0, 1, 0], 1.5, 'q2')

        # Z-axis is blue
        self.view.shape.add_arrow(pa_center, pa_center+pa_array[2], [0, 0, 1], 0.1)
        self.view.shape.add_label(pa_center+pa_array[2], [0, 0, 1], 1.5, 'q3')

    def _add_coordinate_axes(self):
        """
        Helper method. Add the x, y and z axes at origin to a NGLView.
        """

        # arguments of add_arrow are: start position, end position, color (in RGB), radius of arrow head
        # arguments of add_label are: position, color (in RGB), size, text

        # X-axis is red
        self.view.shape.add_arrow([0, 0, 0], [1, 0, 0], [1, 0, 0], 0.1)
        self.view.shape.add_label([1, 0, 0], [1, 0, 0], 1.5, 'x')

        # Y-axis is green
        self.view.shape.add_arrow([0, 0, 0], [0, 1, 0], [0, 1, 0], 0.1)
        self.view.shape.add_label([0, 1, 0], [0, 1, 0], 1.5, 'y')

        # Z-axis is blue
        self.view.shape.add_arrow([0, 0, 0], [0, 0, 1], [0, 0, 1], 0.1)
        self.view.shape.add_label([0, 0, 1], [0, 0, 1], 1.5, 'z')

    def plot_ith_frame(self, frame_i: int, axes: bool = True, **kwargs):
        """
        Plot i-th frame of self.u, adding to self.view.

        Args:
            - i: index of the frame
            - axes: if True, draw x, y and z axes
        """
        ith_atoms = self.get_ith_frame(frame_i)

        # don't plot frames that would have tiny opacity anyway
        if "opacity" in kwargs.keys() and kwargs["opacity"] < 0.005:
            return self.view

        self.view.add_component(ith_atoms, default_representation=False)
        # the index is there in order to only affect the last added representation
        self.view[-1].add_representation("ball+stick", **kwargs)
        if axes:
            self._add_coordinate_axes()
        return self.view

    def set_color_magnitude(self, all_magnitudes, cmap = None, norm = None):
        if cmap is None:
            cmap = matplotlib.cm.get_cmap("bwr")
        if norm is None:
            norm = matplotlib.colors.CenteredNorm(vcenter=0, halfrange=np.abs(np.min(all_magnitudes)), clip=True)
        scalarMap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        all_hex = []
        for mag in all_magnitudes:
            rgba = scalarMap.to_rgba(mag)
            hex_color = matplotlib.colors.rgb2hex(rgba)
            all_hex.append(hex_color)
        self.current_magnitudes = np.array(all_hex)

    def set_opacity_magnitude(self, all_magnitudes: NDArray):
        # normalize magnitudes between 0 and 1
        min_mag = np.min(all_magnitudes)
        max_mag = np.max(all_magnitudes)
        self.current_opacities = (all_magnitudes - min_mag)/(max_mag - min_mag)


    def _add_optional_representation_parameters(self, my_index: int, colors: list, opacities: list):
        """
        Helper method if you want to plot several view and pass arguments to them.
        """
        kwargs = {}
        # read from self properties is set
        if self.current_magnitudes is not None:
            colors = self.current_magnitudes
        if self.current_opacities is not None:
            opacities = self.current_opacities
        # add to kwargs
        if colors is not None:
            kwargs["color"] = colors[my_index]
        if opacities is not None:
            if type(opacities) == float or type(opacities) == int:
                kwargs["opacity"] = opacities
            else:
                kwargs["opacity"] = opacities[my_index]
        return kwargs


    def plot_frames_sequential(self, list_indices: list, colors: list = None, opacities: list = None):
        """
        Plot several frames of the self.u next to each other. Automatically ngo to next now if you have too
        many frames to display in one row.

        Args:
            - list_indices: a list of integers, each an frame index to be displayed
            - colors: a list of colors (must be same length as list_indices) or None (default)
            - opacities: a list of opacities (must be same length as list_indices) or None (default)
        """
        all_views = []
        for li, list_i in enumerate(list_indices):
            self.fresh_view()
            # add optional parameters
            if colors and len(colors)==len(list_indices):
                forward_index = li
            else:
                forward_index = list_i
            kwargs = self._add_optional_representation_parameters(forward_index, colors, opacities)
            neig_view = self.plot_ith_frame(list_i, **kwargs)
            # this is also important for nice arragement of figures
            neig_view.layout.width = "200px"
            all_views.append(neig_view)

        # sync all views (so that all plots move if you move any)
        sync_all_views(all_views)
        display_all_views(all_views)


    def plot_frames_overlapping(self, list_indices: list, colors: list = None, opacities: list = None, **kwargs_plot):
        """
        Plot several frames of the self.u overlapping.

        Args:
            - list_indices: a list of integers, each an frame index to be displayed
            - colors: a list of colors (must be same length as list_indices) or None (default)
            - opacities: a list of opacities (must be same length as list_indices) or None (default)

        """

        for li, list_i in enumerate(list_indices):
            # add optional parameters
            if colors and len(colors)==len(list_indices):
                forward_index = li
            else:
                forward_index = list_i
            kwargs = self._add_optional_representation_parameters(forward_index, colors, opacities)
            self.plot_ith_frame(list_i, **kwargs, **kwargs_plot)

        return self.view



def sync_all_views(all_views: list):
    for v in all_views:
        v._set_sync_camera(all_views)


def display_all_views(all_views: list):
    # settings that are important so that rows with too many images nicely overflow in the next row
    box = widgets.Box(layout=widgets.Layout(width='100%', display='inline-flex', flex_flow='row wrap'))
    box.overflow_x = 'auto'
    box.children = [i for i in all_views]
    display(box)