from functools import wraps

import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import NDArray
from seaborn import color_palette
import networkx as nx

from molgri.constants import DEFAULT_ALPHAS_3D, TEXT_ALPHAS_3D, DEFAULT_ALPHAS_4D, TEXT_ALPHAS_4D, COLORS, \
    GRID_ALGORITHMS, NAME2SHORT_NAME
from molgri.plotting.abstract import RepresentationCollection, Representation, MultiRepresentationCollection, \
    PanelRepresentationCollection
from molgri.space.analysis import vector_within_alpha
from molgri.space.polytopes import Polytope, IcosahedronPolytope, Cube3DPolytope, second_neighbours, third_neighbours, \
    PolyhedronFromG
from molgri.space.rotobj import SphereGridNDim


class SphereGridPlot(RepresentationCollection):

    def __init__(self, sphere_grid: SphereGridNDim, **kwargs):
        data_name = sphere_grid.get_standard_name(with_dim=True)
        super().__init__(data_name, **kwargs)
        self.sphere_grid = sphere_grid
        if self.sphere_grid.dimensions == 3:
            self.alphas = DEFAULT_ALPHAS_3D
            self.alphas_text = TEXT_ALPHAS_3D
        else:
            self.alphas = DEFAULT_ALPHAS_4D
            self.alphas_text = TEXT_ALPHAS_4D

    def get_possible_title(self):
        return NAME2SHORT_NAME[self.sphere_grid.algorithm_name]

    def make_grid_plot(self, ax=None, fig=None, save=True):
        """Plot the 3D grid plot, for 4D the 4th dimension plotted as color. It always has limits (-1, 1) and equalized
        figure size"""

        rep = Representation(dimension=3, ax=ax, fig=fig, projection="3d", context=self.context,
                             complexity_level=self.complexity_level)

        points = self.sphere_grid.get_grid_as_array()
        if points.shape[1] == 3:
            rep.ax.scatter(*points.T, color="black", s=30)
        else:
            rep.ax.scatter(*points[:, :3].T, c=points[:, 3].T, s=30)  # cmap=plt.hot()

        rep.ax.view_init(elev=10, azim=30)
        rep.set_axis_limits((-1, 1, -1, 1, -1, 1))
        rep.equalize_axes()
        if save:
            self._save_plot_type(rep, "grid")
        return rep

    def make_grid_colored_with_alpha(self, ax=None, fig=None, central_vector: NDArray = None, save=True):

        rep = Representation(dimension=3, ax=ax, fig=fig, projection="3d", context=self.context,
                             complexity_level=self.complexity_level)

        if self.sphere_grid.dimensions != 3:
            print(f"make_grid_colored_with_alpha currently implemented only for 3D systems.")
            return None
        if central_vector is None:
            central_vector = np.zeros((self.sphere_grid.dimensions,))
            central_vector[-1] = 1

        points = self.sphere_grid.get_grid_as_array()

        # plot vector
        rep.ax.scatter(*central_vector, marker="x", c="k", s=30)
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
                rep.ax.scatter(*array_sel_points.T, color=cp[i], s=30)
            already_plotted.extend(selected_points)

        rep.ax.view_init(elev=10, azim=30)
        rep.set_axis_limits((-1, 1, -1, 1, -1, 1))
        rep.equalize_axes()
        if save:
            self._save_plot_type(rep, "colorful_grid")
        return rep

    def make_uniformity_plot(self, ax=None, fig=None, save=True):
        """
        Creates violin plots that are a measure of grid uniformity. A good grid will display minimal variation
        along a range of angles alpha.
        """

        rep = Representation(dimension=2, ax=ax, fig=fig, context=self.context,
                             complexity_level=self.complexity_level)

        df = self.sphere_grid.get_uniformity_df(alphas=self.alphas)
        sns.violinplot(x=df["alphas"], y=df["coverages"], ax=rep.ax, palette=COLORS, linewidth=1, scale="count", cut=0)
        rep.ax.set_xticklabels(self.alphas_text)

        if save:
            self._save_plot_type(rep, "uniformity")
        return rep

    def make_convergence_plot(self, ax=None, fig=None, save=True):
        """
        Creates convergence plots that show how coverages approach optimal values.
        """

        rep = Representation(dimension=2, ax=ax, fig=fig, context=self.context,
                             complexity_level=self.complexity_level)

        df = self.sphere_grid.get_convergence_df(alphas=self.alphas)

        sns.lineplot(x=df["N"], y=df["coverages"], ax=rep.ax, hue=df["alphas"],
                     palette=color_palette("hls", len(self.alphas_text)), linewidth=1)
        sns.lineplot(x=df["N"], y=df["ideal coverage"], style=df["alphas"], ax=rep.ax, color="black")
        rep.ax.set_xscale("log")
        rep.ax.set_yscale("log")
        rep.ax.get_legend().remove()

        if save:
            self._save_plot_type(rep, "convergence")
        return rep

    def make_rot_animation(self):
        plotted_points = self.make_grid_plot(save=False)
        self.animate_figure_view(plotted_points)

    def make_ordering_animation(self):
        representation = self.make_grid_plot(save=False)
        sc = representation.ax.collections[0]

        def update(i):
            current_colors = np.concatenate([facecolors_before[:i], all_white[i:]])
            sc.set_facecolors(current_colors)
            sc.set_edgecolors(current_colors)
            return sc,

        facecolors_before = sc.get_facecolors()
        shape_colors = facecolors_before.shape
        all_white = np.zeros(shape_colors)

        representation.ax.view_init(elev=10, azim=30)
        ani = FuncAnimation(representation.fig, func=update, frames=len(facecolors_before), interval=5, repeat=False)
        self._save_animation_type(ani, "order", fps=3)

    def make_trans_animation(self, ax=None):
        points = self.sphere_grid.get_grid_as_array()
        # create the axis with the right num of dimensions

        dimension = points.shape[1] - 1

        rep = Representation(dimension=dimension, projection="3d", ax=ax, context=self.context,
                             complexity_level=self.complexity_level)

        # sort by the value of the specific dimension you are looking at
        ind = np.argsort(points[:, -1])
        points_3D = points[ind]
        # map the 4th dimension into values 0-1
        alphas = points_3D[:, -1].T
        alphas = (alphas - np.min(alphas)) / np.ptp(alphas)

        # plot the lower-dimensional scatterplot
        all_points = []
        for line in points_3D:
            all_points.append(rep.ax.scatter(*line[:rep.dimension], color="black", alpha=1))

        rep.set_axis_limits((-1, 1) * dimension)
        rep.equalize_axes()

        if len(points) < 20:
            step = 1
        elif len(points) < 100:
            step = 5
        else:
            step = 20

        def animate(frame):
            # plot current point
            current_time = alphas[frame * step]

            for i, p in enumerate(all_points):
                new_alpha = np.max([0, 1 - np.abs(alphas[i] - current_time) * 10])
                p.set_alpha(new_alpha)
            return rep.ax,

        anim = FuncAnimation(rep.fig, animate, frames=len(points) // step,
                             interval=100)  # , frames=180, interval=50
        self._save_animation_type(anim, "trans", fps=len(points) // 5)

    def create_all_plots(self, and_animations=False):
        self.make_grid_plot()
        self.make_grid_colored_with_alpha()
        self.make_uniformity_plot()
        self.make_convergence_plot()
        if and_animations:
            self.make_rot_animation()
            self.make_ordering_animation()
            self.make_trans_animation()


class PolytopePlot(RepresentationCollection):

    def __init__(self, polytope: Polytope, **kwargs):
        self.polytope = polytope
        split_name = str(polytope).split()
        data_name = f"{split_name[0]}_{split_name[-1]}"
        super().__init__(data_name, **kwargs)

    def make_graph(self, ax=None, fig=None, with_labels=True, save=True):
        """
        Plot the networkx graph of self.G.
        """

        rep = Representation(2, fig=fig, ax=ax, context=self.context, complexity_level=self.complexity_level,
                             color_style=self.color_style)

        node_labels = {i: tuple(np.round(i, 3)) for i in self.polytope.G.nodes}
        nx.draw_networkx(self.polytope.G, pos=nx.spring_layout(self.polytope.G, weight="p_dist"),
                         with_labels=with_labels, labels=node_labels)
        if save:
            self._save_plot_type(rep, "graph")

    def make_neighbours_plot(self, ax = None, fig=None, node_i=0):
        """
        Want to see which points count as neighbours, second- or third neighbours of a specific node? Use this plotting
        method.
        """

        if self.polytope.d != 3:
            print(f"Plotting neighbours not available for d={self.polytope.d}")
            return

        rep = Representation(3, fig=fig, ax=ax, context=self.context, complexity_level=self.complexity_level,
                             color_style=self.color_style)

        all_nodes = self.polytope.get_node_coordinates()
        node = tuple(all_nodes[node_i])
        neig = self.polytope.G.neighbors(node)
        sec_neig = list(second_neighbours(self.polytope.G, node))
        third_neig = list(third_neighbours(self.polytope.G, node))
        for sel_node in all_nodes:

            rep.ax.scatter(*sel_node, color="black", s=30, alpha=0.5)
            if np.allclose(sel_node, node):
                rep.ax.scatter(*sel_node, color="red", s=40, alpha=0.5)
            if tuple(sel_node) in neig:
                rep.ax.scatter(*sel_node, color="blue", s=38, alpha=0.5)
            if tuple(sel_node) in sec_neig:
                rep.ax.scatter(*sel_node, color="green", s=35, alpha=0.5)
            if tuple(sel_node) in third_neig:
                rep.ax.scatter(*sel_node, color="orange", s=32, alpha=0.5)
        self.save_plot(name_addition=f"neighbours_{node_i}")

    def make_node_plot(self, ax = None, select_faces: set = None, projection: bool = False, plot_edges=False,
                       color_by="level"):
        """
        Plot the points of the polytope + possible division points. Colored by level at which the point was added.
        Or: colored by index to see how sorting works. Possible to select only one or a few faces on which points
        are to be plotted for clarity.

        Args:
            ax: axis
            select_faces: a set of face numbers that can range from 0 to number of faces of the polyhedron, e.g. {0, 5}.
                          If None, all faces are shown.
            projection: True if you want to plot the projected points, not the ones on surfaces of polytope
            color_by: "level" or "index"
        """
        if self.polytope.d > 3:
            print(f"Plotting nodes not available for d={self.polytope.d}")
            return

        self._create_fig_ax(ax, dim=3, projection="3d")
        self._set_up_empty()
        level_color = ["black", "red", "blue", "green"]
        index_palette = color_palette("coolwarm", n_colors=self.polytope.G.number_of_nodes())

        for i, point in enumerate(self.polytope.get_N_ordered_points(projections=False)):
            # select only points that belong to at least one of the chosen select_faces (or plot all if None selection)
            node = self.polytope.G.nodes[tuple(point)]
            point_faces = set(node["face"])
            point_level = node["level"]
            point_projection = node["projection"]
            if select_faces is None or len(point_faces.intersection(select_faces)) > 0:
                # color selected based on the level of the node or index of the sorted nodes
                if color_by == "level":
                    color = level_color[point_level]
                elif color_by == "index":
                    color = index_palette[i]
                else:
                    raise ValueError(f"The argument color_by={color_by} not possible (try 'index', 'level')")

                if projection:
                    self.ax.scatter(*point_projection, color=color, s=30)
                else:
                    self.ax.scatter(*point, color=color, s=30)
                    self.ax.text(*point, s=f"{i}")
        if plot_edges:
            self._plot_edges(self.ax, select_faces=select_faces)
        self.save_plot(name_addition=f"points_{color_by}")

    def _plot_edges(self, ax, select_faces=None, label=None, **kwargs):
        """
        Plot the edges between the points. Can select to display only some faces for clarity.

        Args:
            ax: axis
            select_faces: a set of face numbers from 0 to (incl) 19, e.g. {0, 5}. If None, all faces are shown.
            label: select the name of edge parameter if you want to display it
            **kwargs: other plotting arguments
        """
        for edge in self.polytope.G.edges(data=True):
            faces_edge_1 = set(self.polytope.G.nodes[edge[0]]["face"])
            faces_edge_2 = set(self.polytope.G.nodes[edge[1]]["face"])
            # both the start and the end point of the edge must belong to one of the selected faces
            n1_on_face = select_faces is None or len(faces_edge_1.intersection(select_faces)) > 0
            n2_on_face = select_faces is None or len(faces_edge_2.intersection(select_faces)) > 0
            if n1_on_face and n2_on_face:
                # usually you only want to plot edges used in division
                ax.plot(*np.array(edge[:2]).T, color="black",  **kwargs)
                if label:
                    midpoint = np.average(np.array(edge[:2]), axis=0)
                    s = edge[2][f"{label}"]
                    ax.text(*midpoint, s=f"{s:.3f}")

    def make_cell_plot(self, ax = None, cell_index: int = 0, draw_edges: bool = True):
        """
        Since you cannot visualise a 4D object directly, here's an option to visualise the 3D sub-cells of a 4D object.

        Args:
            ax: axis
            cell_index: index of the sub-cell to plot (in cube4D that can be 0-7)
            draw_edges: use True if you want to also draw edges, False if only points
        """
        if self.polytope.d != 4:
            print(f"Plotting cells not available for d={self.polytope.d}")
            return

        self._create_fig_ax(ax, dim=3, projection="3d")
        self._set_up_empty()
        # find the points that belong to the chosen cell_index
        nodes = (
            node
            for node, data
            in self.polytope.G.nodes(data=True)
            if cell_index in data.get('face')
        )
        subgraph = self.polytope.G.subgraph(nodes)
        # find the component corresponding to the constant 4th dimension
        arr_nodes = np.array(subgraph.nodes)
        dim_to_keep = list(np.where(~np.all(arr_nodes == arr_nodes[0, :], axis=0))[0])
        new_nodes = {old: (old[dim_to_keep[0]], old[dim_to_keep[1]], old[dim_to_keep[2]]) for old in subgraph.nodes}
        subgraph_3D = nx.relabel_nodes(subgraph, new_nodes)
        # create a 3D polyhedron and use its plotting functions
        sub_polyhedron = PolyhedronFromG(subgraph_3D)
        poly_plotter = PolytopePlot(sub_polyhedron)
        poly_plotter.make_node_plot(ax=self.ax, plot_edges=draw_edges)
        self.save_plot(name_addition=f"cell")

    def create_all_plots(self):
        self.make_graph(with_labels=True)
        self.make_neighbours_plot()
        self.make_node_plot(projection=True, color_by="level")
        self.make_node_plot(plot_edges=True, color_by="index")
        self.make_node_plot(select_faces={0, 1, 2})
        self.make_cell_plot()


class PanelSphereGridPlots(PanelRepresentationCollection):

    def __init__(self, N_points: int, grid_dim: int, default_context = None, default_complexity_level = None,
                 default_color_style=None, **kwargs):
        list_plots = []
        for alg in GRID_ALGORITHMS[:-1]:
            sphere_grid = SphereGridFactory.create(alg_name=alg, N=N_points, dimensions=grid_dim, print_messages=False,
                                                   time_generation=False, use_saved=False)
            sphere_plot = SphereGridPlot(sphere_grid, default_context=default_context,
                                         default_color_style=default_color_style,
                                         default_complexity_level=default_complexity_level)
            list_plots.append(sphere_plot)
        data_name = f"all_{N_points}"
        super().__init__(data_name, list_plots, **kwargs)

    def make_all_grid_plots(self):
        fig, all_ax, reprs = self.make_plot_for_all(3, "make_grid_plot")
        self._save_multiplot(fig, "grid")

    def make_all_convergence_plots(self):
        fig, all_ax, reprs = self.make_plot_for_all(2, "make_convergence_plot")
        self._save_multiplot(fig, "convergence")

    def make_all_uniformity_plots(self):
        fig, all_ax, reprs = self.make_plot_for_all(2, "make_uniformity_plot")
        self._save_multiplot(fig, "uniformity")


if __name__ == "__main__":
    from molgri.space.rotobj import SphereGridFactory
    sgf_ico = SphereGridFactory.create(alg_name="ico", N=200, dimensions=4,
                                   print_messages=False, time_generation=False,
                                   use_saved=False)
    sp1 = SphereGridPlot(sgf_ico, default_context="talk", default_complexity_level="half_empty")

    sp1.create_all_plots(and_animations=True)

    psgp = PanelSphereGridPlots(200, grid_dim=4, default_context="talk", default_complexity_level="half_empty")
    psgp.make_all_grid_plots()
    psgp.make_all_uniformity_plots()
    psgp.make_all_convergence_plots()
