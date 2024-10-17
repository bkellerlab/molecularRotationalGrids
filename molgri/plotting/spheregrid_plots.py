"""
Plotting SphereGridNDim and Polytope-based objects.

Visualising 3D with 3D and hammer plots and 4D with translation animation and cell plots. Plotting spherical Voronoi
cells and their areas. A lot of convergence and uniformity testing.
"""
from copy import copy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist

from molgri.space.utils import distance_between_quaternions, find_inverse_quaternion, points4D_2_8cells, which_row_is_k
from molgri.paths import PATH_OUTPUT_PLOTS
from numpy.typing import NDArray, ArrayLike
from seaborn import color_palette
import networkx as nx
import matplotlib as mpl

from molgri.constants import DEFAULT_ALPHAS_3D, TEXT_ALPHAS_3D, \
    DEFAULT_ALPHAS_4D, TEXT_ALPHAS_4D, NAME2SHORT_NAME
from molgri.plotting.abstract import (ArrayPlot, MultiRepresentationCollection, RepresentationCollection)
from molgri.space.polytopes import Polytope, find_opposing_q, second_neighbours, third_neighbours, Cube4DPolytope
from molgri.space.rotobj import SphereGridNDim, SphereGrid4DFactory, \
    ConvergenceSphereGridFactory
from molgri.wrappers import plot3D_method, plot_method


class SphereGridPlot(ArrayPlot):

    """
    For plotting SphereGridNDim objects and their properties.
    """

    def __init__(self, sphere_grid: SphereGridNDim, **kwargs):
        data_name = sphere_grid.get_name(with_dim=True)
        super().__init__(data_name, my_array=sphere_grid.get_grid_as_array(),
                         default_axes_limits=(-1, 1, -1, 1, -1, 1), **kwargs)
        self.sphere_grid = sphere_grid
        if self.sphere_grid.dimensions == 3:
            self.alphas = DEFAULT_ALPHAS_3D
            self.alphas_text = TEXT_ALPHAS_3D
        else:
            self.alphas = DEFAULT_ALPHAS_4D
            self.alphas_text = TEXT_ALPHAS_4D

    def get_possible_title(self) -> str:
        """
        Title of the plot - useful for MultiPlots.
        """
        return NAME2SHORT_NAME[self.sphere_grid.algorithm_name]



    @plot_method
    def plot_cdist_array(self, only_upper=False):
        # TODO: all distances (not only neighbours)
        grid = self.sphere_grid.get_grid_as_array(only_upper=only_upper)
        vmax = self.sphere_grid.get_center_distances(only_upper=only_upper).max()
        if self.sphere_grid.dimensions == 3:
            metric="cos"
        else:
            metric = distance_between_quaternions
        cmap = copy(plt.cm.get_cmap('plasma'))
        cmap.set_over("black")
        center_distances_array = cdist(grid, grid, metric=metric)
        sns.heatmap(center_distances_array, cmap=cmap, ax=self.ax, vmax=vmax)
        self._equalize_axes()

    def _plot_8cells(self, points_4D: NDArray, labels=True, color="black", alpha: float = 1.0):
        eight_lists, eight_indices = points4D_2_8cells(points_4D)

        for i, subplot in enumerate(self.ax.ravel()):
            if eight_lists[i]:
                subplot.scatter(*np.array(eight_lists[i]).T, color=color, alpha=alpha)
                if labels:
                    for j, el in enumerate(eight_lists[i]):
                        subplot.text(*el, f"{eight_indices[i][j]}")
        self._set_axis_limits((-1, 1, -1, 1, -1, 1))
        self._equalize_axes()

    @plot_method
    def plot_8comp(self, only_upper=False, labels=True):
        if self.sphere_grid.dimensions != 4:
            print("Function only available for hypersphere grids")
            return
        self._create_fig_ax(num_columns=4, num_rows=2, projection="3d", figsize=(20, 10),
                            complexity="half_empty")
        points = self.sphere_grid.get_grid_as_array(only_upper=only_upper)
        self._plot_8cells(points, labels=labels, color="black", alpha=1)

    @plot_method
    def plot_8comp_neighbours(self, point_index: int = 0, only_upper=False, labels=True,
                              include_opposing_neighbours=True):
        if self.sphere_grid.dimensions != 4:
            print("Function only available for hypersphere grids")
            return
        self._create_fig_ax(num_columns=4, num_rows=2, projection="3d", figsize=(20, 10),
                            complexity="half_empty")
        points = self.sphere_grid.get_grid_as_array(only_upper=only_upper)
        self._plot_8cells(points, labels=labels, color="black", alpha=0.5)
        # the point itself
        self._plot_8cells(np.array([points[point_index],]), labels=False, color="red")
        # neighbours
        neighbours = self.sphere_grid.get_voronoi_adjacency(only_upper=only_upper,
                                                            include_opposing_neighbours=include_opposing_neighbours).toarray()
        self._plot_8cells(points[np.nonzero(neighbours[point_index])[0]], labels=False, color="blue")
        # opposite point
        if include_opposing_neighbours:
            opp_index = which_row_is_k(points, find_inverse_quaternion(points[point_index]))
            if len(opp_index) == 1:
                self._plot_8cells(np.array([points[opp_index[0]],]), labels=False, color="orange")

    @plot_method
    def plot_8comp_voronoi(self, points = True, vertices=True, only_upper=False, labels=True):
        if self.sphere_grid.dimensions != 4:
            print("Function only available for hypersphere grids")
            return
        self._create_fig_ax(num_columns=4, num_rows=2, projection="3d", figsize=(20, 10),
                            complexity="half_empty")
        if points:
            points = self.sphere_grid.get_grid_as_array(only_upper=only_upper)
            self._plot_8cells(points, labels=labels, color="black", alpha=1)
        if vertices:
            vertices = self.sphere_grid.get_sv_vertices(only_upper=only_upper)
            self._plot_8cells(vertices, labels=labels, color="green", alpha=1)


class PolytopePlot(RepresentationCollection):

    """
    This class is for plotting 3D polytopes, some methods may also be suitable for hypercubes, but for them see also
    EightCellsPlot.
    """

    def __init__(self, polytope: Polytope, **kwargs):
        self.polytope = polytope
        split_name = str(polytope).split()
        data_name = f"{split_name[0]}_{split_name[-1]}"
        default_complexity_level = kwargs.pop("default_complexity_level", "half_empty")
        super().__init__(data_name, default_complexity_level=default_complexity_level, **kwargs)

    @plot_method
    def plot_graph(self, with_labels=True):
        """
        Plot the networkx graph of self.G.
        """
        node_labels = {i: tuple(np.round(i, 3)) for i in self.polytope.G.nodes}
        nx.draw_networkx(self.polytope.G, pos=nx.spring_layout(self.polytope.G, weight="p_dist"),
                         with_labels=with_labels, labels=node_labels, ax=self.ax)

    @plot_method
    def plot_cdist(self, **kwargs):
        cdistances = self.polytope.get_cdist_matrix(**kwargs)

        vmin = 0.9 * np.min(cdistances[cdistances > 0.01])

        cmap = mpl.colormaps.get_cmap('gray_r')
        cmap.set_under("blue")

        sns.heatmap(cdistances, cmap=cmap, cbar=False, ax=self.ax, vmin=vmin)
        self.ax.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.set_aspect('equal')
        plt.tight_layout()

    def plot_decomposed_cdist(self, vmax=2, save=True, **kwargs):
        cdist_matrix = self.polytope.get_cdist_matrix(**kwargs)
        # show components individually
        vmin = 0.9 * np.min(cdist_matrix[cdist_matrix > 1e-5])

        individual_values = set(list(np.round(cdist_matrix.flatten(), 6)))
        individual_values = sorted([x for x in individual_values if vmin < x < vmax])
        num_values = len(individual_values)

        if num_values > 5:
            num_columns = num_values // 2 + 1
            num_rows = 2
        else:
            num_columns = num_values
            num_rows = 1
        self._create_fig_ax(num_columns=num_columns, num_rows=num_rows)

        cum_num_neig = np.zeros(len(cdist_matrix))

        decompositions = []

        for i, value in enumerate(individual_values):
            subax = self.ax.ravel()[i]
            mask = np.isclose(cdist_matrix, value)
            sns.heatmap(mask, ax=subax, cmap="Reds", cbar=False)
            decompositions.append(mask)

            subax.set_aspect('equal')
            subax.get_xaxis().set_visible(False)
            subax.axes.get_yaxis().set_visible(False)
            cum_num_neig += np.sum(mask, axis=0)
            subax.set_title(f"Dist={str(np.round(value, 3))}, CN={np.round(np.average(cum_num_neig), 3)}",
                            fontsize=20)
        plt.tight_layout()
        if save:
            plt.savefig(PATH_OUTPUT_PLOTS + f"decomposed_cdist")
        return decompositions

    @plot_method
    def plot_adj_matrix(self, exclude_nans=True, **kwargs):
        adj_matrix = self.polytope.get_polytope_adj_matrix(**kwargs).toarray()
        cmap = mpl.colormaps.get_cmap('gray')
        cmap.set_bad("green")

        if exclude_nans:
            # Find the indices of rows and columns that have no NaN values
            valid_rows = np.all(~np.isnan(adj_matrix), axis=1)
            valid_columns = np.all(~np.isnan(adj_matrix), axis=0)

            # Extract the valid rows and columns from the original array
            extracted_arr = adj_matrix[valid_rows, :]
            extracted_arr = extracted_arr[:, valid_columns]
            adj_matrix = extracted_arr
        sns.heatmap(adj_matrix, cmap=cmap, ax=self.ax, cbar=False)
        self.ax.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Avg neighbours={np.average(np.sum(adj_matrix, axis=0))}")
        plt.tight_layout()

    @plot3D_method
    def plot_neighbours(self, node_i: int = 0, up_to: int = 2, edges=False, projected=False):
        """
        Want to see which points count as neighbours, second- or third neighbours of a specific node? Use this plotting
        method.

        Args:
            up_to: 1, 2 or 3 -> plot up to first, second or third neighbours.
        """

        if up_to > 3:
            print("Cannot plot more than third neighbours. Will proceed with third neighbours")
            up_to = 3

        # in black plot all nodes and straight edges
        self.plot_nodes(ax=self.ax, fig=self.fig, save=False, color_by=None, plot_edges=edges, projected=projected)
        self.plot_edges(ax=self.ax, fig=self.fig, save=False, edge_categories=[0])

        all_nodes = self.polytope.get_nodes(projection=projected)

        # plot the selected node in red
        node = tuple(all_nodes[node_i])
        self.plot_single_points([node, ], color="red", ax=self.ax, fig=self.fig, label=True, save=False)

        # plot first neighbours
        neig = self.polytope.G.neighbors(tuple(node))
        for n in neig:
            n_index = which_row_is_k(all_nodes, n)[0]
            self.plot_single_points([all_nodes[n_index]], color="blue", ax=self.ax, fig=self.fig,
                                    label=True, save=False)

        # optionally plot second and third neighbours
        if up_to >= 2:
            sec_neig = second_neighbours(self.polytope.G, node)
            for n in sec_neig:
                n_index = which_row_is_k(all_nodes, n)[0]
                self.plot_single_points([all_nodes[n_index]], color="green", ax=self.ax, fig=self.fig,
                                        label=True, save=False)
        if up_to == 3:
            third_neig = third_neighbours(self.polytope.G, node)
            for n in third_neig:
                n_index = which_row_is_k(all_nodes, n)[0]
                self.plot_single_points([all_nodes[n_index]], color="orange", ax=self.ax, fig=self.fig,
                                        label=True, save=False)

    @plot3D_method
    def plot_nodes(self, select_faces: set = None, projected: bool = False, plot_edges: bool = False,
                   plot_nodes: bool = True, color_by: str = "level", label=False, N=None, edge_categories=None):
        """
        Plot the points of the polytope + possible division points. Colored by level at which the point was added.
        Or: colored by index to see how sorting works. Possible to select only one or a few faces on which points
        are to be plotted for clarity.

        Args:
            fig: figure
            ax: axis
            save: whether to save fig
            select_faces: a set of face numbers that can range from 0 to number of faces of the polyhedron, e.g. {0, 5}.
                          If None, all faces are shown.
            projected: True if you want to plot the projected points, not the ones on surfaces of polytope
            plot_edges: select True if you want to see connections between nodes
            color_by: "level" or "index"
        """

        if edge_categories is None:
            edge_categories = [0]

        nodes_poly = self.polytope.get_nodes(N=N, projection=False)

        level_color = ["black", "red", "blue", "green"]
        index_palette = color_palette("coolwarm", n_colors=self.polytope.G.number_of_nodes())

        if plot_nodes:
            for i, node in enumerate(nodes_poly):
                # select only points that belong to at least one of the chosen select_faces (or plot all if None)
                point_faces = set(self.polytope.G.nodes[tuple(node)]["face"])
                point_level = self.polytope.G.nodes[tuple(node)]["level"]
                if select_faces is None or len(point_faces.intersection(select_faces)) > 0:
                    # color selected based on the level of the node or index of the sorted nodes
                    if color_by == "level":
                        color = level_color[point_level]
                    elif color_by == "index":
                        color = index_palette[i]
                    elif color_by is None:
                        color = "black"
                    else:
                        raise ValueError(f"The argument color_by={color_by} not possible (try 'index', 'level')")
                    self.plot_single_points([node, ], color=color, label=label, ax=self.ax,
                                            fig=self.fig, save=False, projected=projected)
        self._set_axis_limits((-0.6, 0.6, -0.6, 0.6, -0.6, 0.6))
        self._equalize_axes()
        if plot_edges:
            self.plot_edges(nodes=[tuple(n) for n in nodes_poly], select_faces=select_faces, ax=self.ax, fig=self.fig,
                            edge_categories=edge_categories, save=False)

    @plot3D_method
    def plot_single_points(self, nodes: ArrayLike = None, color ="black", label=True, projected=False):
        """
        Helper function that should be called whenever plotting any (list of) nodes.

        Don't plot anything (but don't return an error) if the specified node doesn't exist.
        """
        if nodes is None:
            return

        if self.polytope.d > 3:
            print("Plotting nodes not available for d> 3")
            return

        for i, n in enumerate(nodes):
            ci = self.polytope.G.nodes[tuple(n)]["central_index"]

            if projected:
                to_plot = self.polytope.G.nodes[tuple(n)]["projection"]
            else:
                to_plot = n
            self.ax.scatter(*to_plot, color=color, s=30)
            if label:
                self.ax.text(*to_plot, ci)

        self._equalize_axes()

    @plot3D_method
    def plot_edges(self, nodes: ArrayLike = None, select_faces=None, label=None,
                   edge_categories=None, **kwargs):
        """
        Helper function that should be called whenever plotting any (list of) edges. Can select to display only some
        faces for clarity.

        Args:
            select_faces: a set of face numbers from 0 to (incl) 19, e.g. {0, 5}. If None, all faces are shown.
            label: select the name of edge parameter if you want to display it
            **kwargs: other plotting arguments
        """
        if nodes is None:
            return

        if self.polytope.d > 3:
            print("Plotting nodes not available for d> 3")
            return

        all_edges = self.polytope.get_edges_of_categories(nbunch=nodes, data=True, categories=edge_categories)

        for edge in all_edges:
            faces_edge_1 = set(self.polytope.G.nodes[edge[0]]["face"])
            faces_edge_2 = set(self.polytope.G.nodes[edge[1]]["face"])
            # both the start and the end point of the edge must belong to one of the selected faces
            n1_on_face = select_faces is None or len(faces_edge_1.intersection(select_faces)) > 0
            n2_on_face = select_faces is None or len(faces_edge_2.intersection(select_faces)) > 0
            if n1_on_face and n2_on_face:  # and edge[0] in nodes and edge[1] in nodes
                # usually you only want to plot edges used in division
                self.ax.plot(*np.array(edge[:2]).T, color="black",  **kwargs)
                if label:
                    midpoint = np.average(np.array(edge[:2]), axis=0)
                    s = edge[2][f"{label}"]
                    self.ax.text(*midpoint, s=f"{s:.3f}")


class EightCellsPlot(MultiRepresentationCollection):

    def __init__(self, cube4D: Cube4DPolytope, only_half_of_cube=True):
        self.cube4D = cube4D
        self.only_half_of_cube = only_half_of_cube
        if only_half_of_cube:
            all_subpolys = cube4D.get_all_cells(include_only=cube4D.get_half_of_hypercube())
        else:
            all_subpolys = cube4D.get_all_cells()
        list_plots: List[PolytopePlot] = [PolytopePlot(subpoly) for subpoly in all_subpolys]
        super().__init__(data_name=f"eight_cells_{cube4D.current_level}_{only_half_of_cube}",
                         list_plots=list_plots, n_rows=2, n_columns=4)

    def make_all_eight_cells(self, animate_rot=False, save=True, **kwargs):
        if "color_by" not in kwargs.keys():
            kwargs["color_by"] = None
        if "plot_edges" not in kwargs.keys():
            kwargs["plot_edges"] = True
        self._make_plot_for_all("plot_nodes", projection="3d", plotting_kwargs=kwargs)
        if animate_rot:
            return self.animate_figure_view("points", dpi=100)
        if save:
            self._save_multiplot("points", dpi=100)

    def _plot_in_color(self, nodes, color, **kwargs):
        self._make_plot_for_all("_plot_single_points",
                                plotting_kwargs={"nodes": nodes, "color": color, "label": True},
                                projection="3d", **kwargs)

    def make_all_8cell_neighbours(self, node_index: int = 15, save=True, animate_rot=False,
                                  include_opposing_neighbours=True):

        neighbours_indices = self.cube4D.get_neighbours_of(node_index,
                                                           include_opposing_neighbours=include_opposing_neighbours,
                                                           only_half_of_cube=self.only_half_of_cube)
        if include_opposing_neighbours:
            node = self.cube4D.get_nodes()[node_index]
            opp_node = find_opposing_q(tuple(node), self.cube4D.G)
            opp_node_index = self.cube4D.G.nodes[opp_node]["central_index"]

        self.make_all_eight_cells(save=False, plot_edges=True, edge_categories=[0])

        for i, subax in enumerate(self.all_ax.ravel()):
            ci2node = {d["central_index"]:n for n, d in self.list_plots[i].polytope.G.nodes(data=True)}
            # the node itself
            if node_index in ci2node.keys():
                node = ci2node[node_index]
                self.list_plots[i].plot_single_points([tuple(node), ], "red", ax=subax, fig=self.fig,
                                                      save=False)
            # the opposing node:
            if include_opposing_neighbours and opp_node_index in ci2node.keys():
                opp_node_3d = self.list_plots[i].polytope.get_nodes_by_index([opp_node_index,])
                self.list_plots[i].plot_single_points(opp_node_3d, "orange", ax=subax, fig=self.fig,
                                                      save=False)

            # neighbours
            neighbour_nodes = self.list_plots[i].polytope.get_nodes_by_index(neighbours_indices)

            for ni in neighbour_nodes:
                self.list_plots[i].plot_single_points([tuple(ni), ], "blue", ax=subax, fig=self.fig, save=False)
        if animate_rot:
            return self.animate_figure_view(f"neig_{node_index}", dpi=100)
        if save:
            self._save_multiplot(f"neig_{node_index}", dpi=100)


if __name__ == "__main__":
    sphere = SphereGrid4DFactory.create("cube4D", 60, use_saved=False)
    sg = SphereGridPlot(sphere)
    #sg.plot_voronoi(only_upper=False, ax=sg.ax, fig=sg.fig, animate_rot=True, polygons=True)
    #sg.plot_cdist_array()
    #sg.plot_center_distances_array()
    sg.plot_8comp_voronoi(only_upper=False, save=False, labels=False)
    plt.show()
    #sg.plot_8comp_neighbours(only_upper=True)

    # full divisions would be 8, 40, 272
    # hypersphere = SphereGrid4DFactory.create("cube4D", 40, use_saved=False)
    # #print(hypersphere.get_full_hypersphere_array()[0:5], hypersphere.get_full_hypersphere_array()[40:45])
    # # polytope = hypersphere.polytope
    # sg = SphereGridPlot(hypersphere)
    # sg.plot_grid(save=False)
    # sg.plot_voronoi(only_upper=True, ax=sg.ax, fig=sg.fig)
    #
    # #print(len(hypersphere.get_grid_as_array()), len(hypersphere.get_full_hypersphere_array()))
    #
    #
    # ecp = EightCellsPlot(polytope, only_half_of_cube=False)
    # #ecp.make_all_eight_cells(animate_rot=False, label=True)
    # ecp.make_all_8cell_neighbours(0)



