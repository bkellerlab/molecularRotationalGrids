"""
Plotting SphereGridNDim and Polytope-based objects.

Visualising 3D with 3D and hammer plots and 4D with translation animation and cell plots. Plotting spherical Voronoi
cells and their areas. A lot of convergence and uniformity testing.
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from molgri.assertions import which_row_is_k
from molgri.paths import PATH_OUTPUT_PLOTS
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray, ArrayLike
from seaborn import color_palette
import networkx as nx
import matplotlib as mpl

from molgri.constants import DEFAULT_ALPHAS_3D, GRID_ALGORITHMS_3D, GRID_ALGORITHMS_4D, TEXT_ALPHAS_3D, \
    DEFAULT_ALPHAS_4D, TEXT_ALPHAS_4D, COLORS, NAME2SHORT_NAME, DIM_SQUARE
from molgri.plotting.abstract import (ArrayPlot, MultiRepresentationCollection, RepresentationCollection,
                                      plot_voronoi_cells)
from molgri.space.analysis import vector_within_alpha
from molgri.space.polytopes import Polytope, find_opposing_q, second_neighbours, third_neighbours, Cube4DPolytope
from molgri.space.rotobj import SphereGridFactory, SphereGridNDim, SphereGrid3DFactory, SphereGrid4DFactory, \
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

    @plot3D_method
    def plot_alpha_grid(self, central_vector: NDArray = None):
        if self.sphere_grid.dimensions != 3:
            print(f"make_grid_colored_with_alpha currently implemented only for 3D systems.")
            return None
        if central_vector is None:
            central_vector = np.zeros((self.sphere_grid.dimensions,))
            central_vector[-1] = 1

        points = self.sphere_grid.get_grid_as_array()

        # plot vector
        self.ax.scatter(*central_vector, marker="x", c="k", s=30)
        # determine color palette
        cp = sns.color_palette("Spectral", n_colors=len(self.alphas))
        # sort points which point in which alpha area
        already_plotted = []
        for i, alpha in enumerate(self.alphas):
            possible_points = np.array([vec for vec in points if tuple(vec) not in already_plotted])
            within_alpha = vector_within_alpha(central_vector, possible_points, alpha)
            selected_points = [tuple(vec) for i, vec in enumerate(possible_points) if within_alpha[i]]
            array_sel_points = np.array(selected_points)
            if np.any(array_sel_points):
                self.ax.scatter(*array_sel_points.T, color=cp[i], s=30)
            already_plotted.extend(selected_points)

        self.ax.view_init(elev=10, azim=30)
        self._set_axis_limits()
        self._equalize_axes()

    @plot_method
    def plot_uniformity(self):
        """
        Creates violin plots that are a measure of grid uniformity. A good grid will display minimal variation
        along a range of angles alpha.
        """
        df = self.sphere_grid.get_uniformity_df(alphas=self.alphas)
        sns.violinplot(x=df["alphas"], y=df["coverages"], ax=self.ax, palette=COLORS, linewidth=1, scale="count", cut=0)
        try:
            self.ax.set_xticklabels(self.alphas_text)
        except ValueError:
            # not the right number of tick labels, no big issue
            pass

    @plot_method
    def plot_convergence(self):
        """
        Creates convergence plots that show how coverages approach optimal values.
        """
        df = self.sphere_grid.get_convergence_df(alphas=self.alphas)
        sns.lineplot(x=df["N"], y=df["coverages"], ax=self.ax, hue=df["alphas"],
                     palette=color_palette("hls", len(self.alphas_text)), linewidth=1)
        sns.lineplot(x=df["N"], y=df["ideal coverage"], style=df["alphas"], ax=self.ax, color="black")
        self.ax.get_legend().remove()
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

    @plot3D_method
    def plot_voronoi(self, labels=False):
        if self.sphere_grid.dimensions != 3:
            print("make_spherical_voronoi_plot only implemented for 3D grids")
            return

        try:
            sv = self.sphere_grid.get_spherical_voronoi_cells()
            plot_voronoi_cells(sv, self.ax, labels=labels)
        except AttributeError:
            pass

        self.ax.view_init(elev=10, azim=30)
        self._set_axis_limits()
        self._equalize_axes()

    @plot_method
    def plot_cell_volumes(self, approx=False, using_detailed_grid=True):
        voronoi_adjacency = self.sphere_grid.get_cell_volumes(approx=approx, using_detailed_grid=using_detailed_grid)
        sns.violinplot(voronoi_adjacency, ax=self.ax)

    @plot_method
    def plot_adjacency_array(self):
        adjacency_matrix = self.sphere_grid.get_voronoi_adjacency().toarray()
        sns.heatmap(adjacency_matrix, cmap="gray", ax=self.ax)
        self._equalize_axes()

    @plot_method
    def plot_cell_border_array(self):
        center_distances_array = self.sphere_grid.get_cell_borders().toarray()
        sns.heatmap(center_distances_array, cmap="gray", ax=self.ax)
        self._equalize_axes()

    @plot_method
    def plot_center_distances_array(self):
        center_distances_array = self.sphere_grid.get_center_distances().toarray()
        sns.heatmap(center_distances_array, cmap="gray", ax=self.ax)
        self._equalize_axes()


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


class PanelSphereGridPlots(MultiRepresentationCollection):

    def __init__(self, N_points: int, grid_dim: int, default_context: str = None, default_complexity_level: str = None,
                 default_color_style: str = None, use_saved=False, **kwargs):
        list_plots = []
        self.grid_dim = grid_dim
        if grid_dim == 3:
            all_alg = GRID_ALGORITHMS_3D
        else:
            all_alg = GRID_ALGORITHMS_4D[:-1]
        n_rows = 1
        n_columns = len(all_alg)
        for alg in all_alg:
            sphere_grid = SphereGridFactory.create(alg_name=alg, N=N_points, dimensions=grid_dim,
                                                   time_generation=False, use_saved=use_saved)
            sphere_plot = SphereGridPlot(sphere_grid, default_context=default_context,
                                         default_color_style=default_color_style,
                                         default_complexity_level=default_complexity_level)
            list_plots.append(sphere_plot)
        data_name = f"all_{N_points}_{grid_dim}d"
        super().__init__(data_name, list_plots, n_rows=n_rows, n_columns=n_columns, **kwargs)

    def make_all_grid_plots(self, animate_rot=False):
        self._make_plot_for_all("plot_grid", projection="3d")
        self.add_titles(list_titles=[subplot.get_possible_title() for subplot in self.list_plots],
                        pad=-14)
        ani=None
        if animate_rot:
            ani = self.animate_figure_view("grid", dpi=100)
        self._save_multiplot("grid", dpi=100)
        if ani is not None:
            return ani

    def make_all_trans_animations(self, save=True):

        all_points = []
        for plot in self.list_plots:
            all_points.append(plot.sphere_grid.get_grid_as_array())


        dimension = all_points[0].shape[1] - 1

        if dimension == 3:
            projection = "3d"
        else:
            projection = None

        figsize = self.n_columns * DIM_SQUARE[0], self.n_rows * DIM_SQUARE[1]
        self.fig, self.all_ax = plt.subplots(self.n_rows, self.n_columns, subplot_kw={'projection': projection},
                                             figsize=figsize)

        # sort by the value of the specific dimension you are looking at
        all_points_3D = []
        for i, points in enumerate(all_points):
            ind = np.argsort(points[:, -1])
            all_points_3D.append(points[ind])
        # map the 4th dimension into values 0-1
        all_alphas = []
        for points_3D in all_points_3D:
            alphas = points_3D[:, -1].T
            alphas = (alphas - np.min(alphas)) / np.ptp(alphas)
            all_alphas.append(alphas)

        # plot the lower-dimensional scatterplot
        plotted_points = []
        for ax, points3D in zip(np.ravel(self.all_ax), all_points_3D):
            sub_plotted_points = []
            for line in points3D:
                sub_plotted_points.append(ax.scatter(*line[:-1], color="black", alpha=1))
            plotted_points.append(sub_plotted_points)

        self.all_ax[0].set_xlim(-1, 1)
        self.all_ax[0].set_ylim(-1, 1)
        if dimension == 3:
            self.all_ax[0].set_zlim(-1, 1)
        self.unify_axis_limits()
        for ax in np.ravel(self.all_ax):
            ax.set_xticks([])
            ax.set_yticks([])
            if dimension == 3:
                ax.set_zticks([])

        self.add_titles(list_titles=[subplot.get_possible_title() for subplot in self.list_plots],
                        pad=-14)

        if len(all_points_3D[0]) < 20:
            step = 1
        elif len(all_points_3D[0]) < 100:
            step = 5
        else:
            step = 20

        def animate(frame):
            # plot current point
            current_time = alphas[frame * step]
            for ax_alphas, ax_points, ax in zip(all_alphas, plotted_points, np.ravel(self.all_ax)):
                for i, p in enumerate(ax_points):
                    new_alpha = np.max([0, 1 - np.abs(ax_alphas[i] - current_time) * 10])
                    p.set_alpha(new_alpha)
            return self.all_ax,

        anim = FuncAnimation(self.fig, animate, frames=len(all_points_3D[0]) // step,
                             interval=100)  # , frames=180, interval=50
        if save:
            self.save_multianimation(anim, "trans", fps=len(all_points_3D[0]) // step // 2, dpi=200)
        return anim

    def make_all_convergence_plots(self):
        self._make_plot_for_all("plot_convergence")
        self.add_titles(list_titles=[subplot.get_possible_title() for subplot in self.list_plots])
        self.set_log_scale()
        self.unify_axis_limits()
        self._save_multiplot("convergence")

    def make_all_uniformity_plots(self):
        self._make_plot_for_all("plot_uniformity")
        self.add_titles(list_titles=[subplot.get_possible_title() for subplot in self.list_plots])
        self.unify_axis_limits()
        self._save_multiplot("uniformity")


class ConvergenceSphereGridPlot(RepresentationCollection):

    def __init__(self, convergence_sph_grid: ConvergenceSphereGridFactory):
        self.convergence_sph_grid = convergence_sph_grid
        super().__init__(self.convergence_sph_grid.get_name())

    def get_possible_title(self):
        full_name = self.convergence_sph_grid.get_name()
        split_name = full_name.split("_")
        return NAME2SHORT_NAME[split_name[1]]

    @plot_method
    def plot_voronoi_convergence(self):
        try:
            voronoi_df = self.convergence_sph_grid.get_spherical_voronoi_areas()
            sns.lineplot(data=voronoi_df, x="N", y="sph. Voronoi cell area", errorbar="sd", ax=self.ax)
            sns.scatterplot(data=voronoi_df, x="N", y="sph. Voronoi cell area", alpha=0.8, color="black", ax=self.ax,
                            s=1)
            sns.scatterplot(data=voronoi_df, x="N", y="ideal area", color="black", marker="x", ax=self.ax)
        except AttributeError:
            pass

        self.ax.set_xscale("log")

    @plot_method
    def plot_spheregrid_time(self):
        time_df = self.convergence_sph_grid.get_generation_times()
        sns.lineplot(time_df, x="N", y="Time [s]", ax=self.ax)



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
    # full divisions would be 8, 40, 272
    hypersphere = SphereGrid4DFactory.create("cube4D", 8)
    polytope = hypersphere.polytope
    sg = SphereGridPlot(hypersphere)
    #sg.plot_grid(animate_rot=True)

    print(len(hypersphere.get_grid_as_array()), len(hypersphere.get_full_hypersphere_array()))


    ecp = EightCellsPlot(polytope)
    ecp.make_all_eight_cells(animate_rot=True)



