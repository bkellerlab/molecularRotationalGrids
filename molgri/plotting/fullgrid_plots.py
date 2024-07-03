"""
Plots connected to the fullgrid module.

Plot position grids in space, Voronoi cells and their volumes etc.
"""

import numpy as np
import seaborn as sns
from matplotlib import colors

from molgri.constants import NAME2SHORT_NAME
from molgri.plotting.abstract import (RepresentationCollection, plot_voronoi_cells, ArrayPlot)
from molgri.space.fullgrid import FullGrid, ConvergenceFullGridO
from molgri.wrappers import plot3D_method, plot_method


class FullGridPlot(ArrayPlot):

    """
    Plotting centered around FullGrid.
    """

    def __init__(self, full_grid: FullGrid, **kwargs):
        self.full_grid = full_grid
        data_name = self.full_grid.get_name()
        my_array = self.full_grid.position_grid.get_position_grid_as_array()
        super().__init__(data_name, my_array, **kwargs)

    def __getattr__(self, name):
        """ Enable forwarding methods to self.position_grid, so that from FullGrid you can access all properties and
         methods of PositionGrid too."""
        return getattr(self.full_grid, name)

    def get_possible_title(self, algs = True, N_points = False):
        name = ""
        if algs:
            o_name = NAME2SHORT_NAME[self.position_grid.o_rotations.algorithm_name]
            b_name = NAME2SHORT_NAME[self.full_grid.b_rotations.algorithm_name]
            name += f"o = {o_name}, b = {b_name}"
        if N_points:
            N_o = self.position_grid.o_rotations.N
            N_b = self.full_grid.b_rotations.N
            N_t = self.position_grid.t_grid.get_N_trans()
            N_name = f"N_o = {N_o}, N_b = {N_b}, N_t = {N_t}"
            if len(name) > 0:
                N_name = f"; {N_name}"
            name += N_name
        return name

    @plot3D_method
    def plot_positions(self, labels: bool = False, c="black"):
        points = self.position_grid.get_position_grid_as_array()
        cmap = "bwr"
        norm = colors.TwoSlopeNorm(vcenter=0)
        self.ax.scatter(*points.T, c=c, cmap=cmap, norm=norm)

        if labels:
            for i, point in enumerate(points):
                self.ax.text(*point, s=f"{i}")

        self.ax.view_init(elev=10, azim=30)
        self._set_axis_limits()
        self._equalize_axes()

    @plot3D_method
    def plot_position_voronoi(self, plot_vertex_points=True, numbered: bool = False, colors=None):

        origin = np.zeros((3,))

        if numbered:
            points = self.get_position_grid_as_array()
            for i, point in enumerate(points):
                self.ax.text(*point, s=f"{i}")

        try:
            voronoi_disc = self.get_position_voronoi()

            for i, sv in enumerate(voronoi_disc):
                plot_voronoi_cells(sv, self.ax, plot_vertex_points=plot_vertex_points, colors=colors)
                # plot rays from origin to highest level
                if i == len(voronoi_disc)-1:
                    for vertex in sv.vertices:
                        ray_line = np.concatenate((origin[:, np.newaxis], vertex[:, np.newaxis]), axis=1)
                        self.ax.plot(*ray_line, color="black")
        except AttributeError:
            pass

        self.ax.view_init(elev=10, azim=30)
        self._set_axis_limits()
        self._equalize_axes()

    @plot_method
    def plot_position_volumes(self):
        all_volumes = self.get_all_position_volumes()
        sns.violinplot(all_volumes, ax=self.ax)

    def _plot_position_N_N(self, my_array = None, **kwargs):
        sns.heatmap(my_array, cmap="gray", ax=self.ax, **kwargs)
        self._equalize_axes()

    @plot_method
    def plot_position_adjacency(self):
        my_array = self.get_adjacency_of_position_grid().toarray()
        self._plot_position_N_N(my_array, cbar=False)
