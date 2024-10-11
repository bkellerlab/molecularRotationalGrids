"""
Plots connected to the fullgrid module.

Plot position grids in space, Voronoi cells and their volumes etc.
"""

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from scipy.sparse import coo_matrix

pio.templates.default = "simple_white"

WIDTH = 600
HEIGHT = 600


def add_fake_legend(fig, color_dict):
    """
    Adds a fake legend to a Plotly figure using annotations.

    Parameters:
        fig: The Plotly figure object to modify.
        color_dict: Dictionary where keys are legend labels and values are corresponding colors.
    """
    # Set the initial position for the annotations
    x_start = 0.45
    y_start = 1.0
    y_step = 0.05

    # Add an annotation for each entry in the color_dict
    annotations = []
    for i, (label, color) in enumerate(color_dict.items()):
        annotations.append(
            go.layout.Annotation(
                x=x_start,  # Position to the right of the plot
                y=y_start - i * y_step,  # Spacing between legend items
                xref="paper", yref="paper",  # Relative position in the plot
                showarrow=False,  # No arrow
                text=f'<span style="color:{color};">⬤</span> {label}',  # Custom text with colored marker
                font=dict(size=12, color='black'),
                align='left',
                xanchor='left',
            )
        )

    # Add annotations to the figure
    fig.update_layout(annotations=annotations)


def plot_adjacency_array(my_adjacency_array, path_to_save):
    color_continuous_scale = [(0, "white"), (0.5, "red"), (1, "black")]
    fig = go.Figure(go.Heatmap(x=my_adjacency_array.row, y=my_adjacency_array.col, z=my_adjacency_array.data,
                    colorscale=color_continuous_scale, showscale=False, zmin=0, zmax=1))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(autorange="reversed", showticklabels=False, ticks="", showline=True, mirror=True,
                     title="Grid point index")
    fig.update_xaxes( showticklabels=False, ticks="", showline=True, mirror=True,
                      title="Grid point index", side ="top", ticksuffix = "  ")
    # title depends if it is total adjacency or separately orientation and position
    if np.any(np.isclose(my_adjacency_array.data, 0.5)):
        add_fake_legend(fig,{"position neighbours\n(same orientation)": "red", "orientation neighbours\n(same position)": "black"})
        #fig.update_layout(title="Red: position neighbours, black: orientation neighbours")
    fig.write_image(path_to_save, width=WIDTH, height=HEIGHT, scale=2)


def plot_array_heatmap(my_array, path_to_save):
    fig = go.Figure(go.Heatmap(x=my_array.row, y=my_array.col, z=my_array.data))
    fig.update_yaxes(autorange="reversed", showticklabels=False, ticks="", showline=True, mirror=True,
                     title="Grid point index")
    fig.update_xaxes(showticklabels=False, ticks="", showline=True, mirror=True,
                     title="Grid point index", side="top", ticksuffix="  ")
    fig.write_image(path_to_save, width=WIDTH, height=HEIGHT, scale=2)


def plot_violin_position_orientation(my_array, adjacency_position, path_to_save):
    df = pd.DataFrame(data=my_array.data, columns=["data"])
    from_where = []
    # determine whether it comes from position grid
    position_pairs =list(zip(adjacency_position.row, adjacency_position.col))
    for row, col in zip(my_array.row, my_array.col):
        if (row, col) in position_pairs:
            from_where.append("orientation neighbours")
        else:
            from_where.append("position neighbours")
    df['which'] = from_where
    violin1 = go.Violin(
        y=df[df.values == "orientation neighbours"]["data"],
        name='Group 1',  # Name shown in the legend
        line_color='black'
    )

    # Create the second violin plot (group 2) on the right y-axis
    violin2 = go.Violin(
        y=df[df.values == "position neighbours"]["data"],
        name='Group 2',  # Name shown in the legend
        line_color='red',
        yaxis='y2'  # Plot this on the second y-axis
    )

    # Create the figure and add both violin plots
    fig = go.Figure(data=[violin1, violin2])

    # Update layout to add a second y-axis
    fig.update_layout(
        yaxis=dict(
            title='Orientation neighbours (same position)',
            showline=True,
            tickfont=dict(color='black'),
            titlefont=dict(color='black'),
            ticks="",
        ),
        yaxis2=dict(
            title='Position neighbours (same orientation)',
            overlaying='y',  # Overlay this y-axis on the same plot
            side='right',  # Place it on the right
            showline=True,
            ticks="",
            tickfont=dict(color='red'),
            titlefont=dict(color='red'),
        ),
    )
    # TODO: remove legend, proper x labels
    fig.write_image(path_to_save, width=WIDTH, height=HEIGHT, scale=2)

# import numpy as np
# import seaborn as sns
# from matplotlib import colors
#
# from molgri.constants import NAME2SHORT_NAME
# from molgri.plotting.abstract import (plot_voronoi_cells, ArrayPlot)
# from molgri.space.fullgrid import FullGrid
# from molgri.wrappers import plot3D_method, plot_method
#
#
# class FullGridPlot(ArrayPlot):
#
#     """
#     Plotting centered around FullGrid.
#     """
#
#     def __init__(self, full_grid: FullGrid, **kwargs):
#         self.full_grid = full_grid
#         data_name = self.full_grid.get_name()
#         my_array = self.full_grid.position_grid.get_position_grid_as_array()
#         super().__init__(data_name, my_array, **kwargs)
#
#     def __getattr__(self, name):
#         """ Enable forwarding methods to self.position_grid, so that from FullGrid you can access all properties and
#          methods of PositionGrid too."""
#         return getattr(self.full_grid, name)
#
#     def get_possible_title(self, algs = True, N_points = False):
#         name = ""
#         if algs:
#             o_name = NAME2SHORT_NAME[self.position_grid.o_rotations.algorithm_name]
#             b_name = NAME2SHORT_NAME[self.full_grid.b_rotations.algorithm_name]
#             name += f"o = {o_name}, b = {b_name}"
#         if N_points:
#             N_o = self.position_grid.o_rotations.N
#             N_b = self.full_grid.b_rotations.N
#             N_t = self.position_grid.t_grid.get_N_trans()
#             N_name = f"N_o = {N_o}, N_b = {N_b}, N_t = {N_t}"
#             if len(name) > 0:
#                 N_name = f"; {N_name}"
#             name += N_name
#         return name
#
#     @plot3D_method
#     def plot_positions(self, labels: bool = False, c="black"):
#         points = self.position_grid.get_position_grid_as_array()
#         cmap = "bwr"
#         norm = colors.TwoSlopeNorm(vcenter=0)
#         self.ax.scatter(*points.T, c=c, cmap=cmap, norm=norm)
#
#         if labels:
#             for i, point in enumerate(points):
#                 self.ax.text(*point, s=f"{i}")
#
#         self.ax.view_init(elev=10, azim=30)
#         self._set_axis_limits()
#         self._equalize_axes()
#
#     @plot3D_method
#     def plot_position_voronoi(self, plot_vertex_points=True, numbered: bool = False, colors=None):
#
#         origin = np.zeros((3,))
#
#         if numbered:
#             points = self.get_position_grid_as_array()
#             for i, point in enumerate(points):
#                 self.ax.text(*point, s=f"{i}")
#
#         try:
#             voronoi_disc = self.get_position_voronoi()
#
#             for i, sv in enumerate(voronoi_disc):
#                 plot_voronoi_cells(sv, self.ax, plot_vertex_points=plot_vertex_points, colors=colors)
#                 # plot rays from origin to highest level
#                 if i == len(voronoi_disc)-1:
#                     for vertex in sv.vertices:
#                         ray_line = np.concatenate((origin[:, np.newaxis], vertex[:, np.newaxis]), axis=1)
#                         self.ax.plot(*ray_line, color="black")
#         except AttributeError:
#             pass
#
#         self.ax.view_init(elev=10, azim=30)
#         self._set_axis_limits()
#         self._equalize_axes()
#
#     @plot_method
#     def plot_position_volumes(self):
#         all_volumes = self.get_all_position_volumes()
#         sns.violinplot(all_volumes, ax=self.ax)
#
#     def _plot_position_N_N(self, my_array = None, **kwargs):
#         sns.heatmap(my_array, cmap="gray", ax=self.ax, **kwargs)
#         self._equalize_axes()
#
#     @plot_method
#     def plot_position_adjacency(self):
#         my_array = self.get_adjacency_of_position_grid().toarray()
#         self._plot_position_N_N(my_array, cbar=False)
