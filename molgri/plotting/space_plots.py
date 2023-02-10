import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import NDArray
from seaborn import color_palette
import networkx as nx

from molgri.constants import DEFAULT_ALPHAS_3D, TEXT_ALPHAS_3D, DEFAULT_ALPHAS_4D, TEXT_ALPHAS_4D, COLORS
from molgri.paths import PATH_OUTPUT_PLOTS
from molgri.plotting.abstract import AbstractPlot, Plot3D
from molgri.space.analysis import vector_within_alpha
from molgri.space.polytopes import Polytope, IcosahedronPolytope, Cube3DPolytope, second_neighbours, third_neighbours, \
    PolyhedronFromG
from molgri.space.rotobj import SphereGridNDim


class SphereGridPlot(AbstractPlot):

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

    def make_grid_plot(self, ax = None):
        """Plot the 3D grid plot, for 4D the 4th dimension plotted as color. It always has limits (-1, 1) and equalized
        figure size"""
        self._create_fig_ax(ax, dim=3, projection="3d")
        self._set_up_empty()
        points = self.sphere_grid.get_grid_as_array()
        if points.shape[1] == 3:
            sc = self.ax.scatter(*points.T, color="black", s=30)
        else:
            sc = self.ax.scatter(*points[:, :3].T, c=points[:, 3].T, s=30)  # cmap=plt.hot()
        self._axis_limits(-1, 1, -1, 1, -1, 1)
        self._equalize_axes()
        self.save_plot(name_addition="grid")
        return sc

    def make_grid_colored_with_alpha(self, ax=None, central_vector: NDArray = None):
        if self.sphere_grid.dimensions != 3:
            print(f"make_grid_colored_with_alpha currently implemented only for 3D systems.")
            return
        if central_vector is None:
            central_vector = np.zeros((self.sphere_grid.dimensions,))
            central_vector[-1] = 1
        points = self.sphere_grid.get_grid_as_array()
        self._create_fig_ax(ax, dim=3, projection="3d")
        self._set_up_empty()
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
                sc = self.ax.scatter(*array_sel_points.T, color=cp[i], s=30)
            already_plotted.extend(selected_points)
        self.ax.view_init(elev=10, azim=30)
        self._axis_limits(-1, 1, -1, 1, -1, 1)
        self._equalize_axes()
        self.save_plot(name_addition="colorful_grid")

    def make_alpha_plot(self, ax = None):
        """
        Creates violin plots that are a measure of grid uniformity. A good grid will display minimal variation
        along a range of angles alpha.
        """
        self._create_fig_ax(ax, dim=2)
        self._set_up_empty()

        df = self.sphere_grid.get_uniformity_df(alphas=self.alphas)
        sns.violinplot(x=df["alphas"], y=df["coverages"], ax=self.ax, palette=COLORS, linewidth=1, scale="count", cut=0)
        self.ax.set_xticklabels(self.alphas_text)
        self.save_plot(name_addition="uniformity")

    def make_convergence_plot(self, ax = None):
        """
        Creates convergence plots that show how coverages approach optimal values.
        """
        self._create_fig_ax(ax, dim=2)
        self._set_up_empty()

        df = self.sphere_grid.get_convergence_df(alphas=self.alphas)

        sns.lineplot(x=df["N"], y=df["coverages"], ax=self.ax, hue=df["alphas"],
                     palette=color_palette("hls", len(self.alphas_text)), linewidth=1)
        sns.lineplot(x=df["N"], y=df["ideal coverage"], style=df["alphas"], ax=self.ax, color="black")
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.get_legend().remove()
        self.save_plot(name_addition="convergence")


    def make_rot_animation(self):
        self.make_grid_plot()
        self.animate_figure_view()

    def make_ordering_animation(self):
        sc = self.make_grid_plot()

        def update(i):
            current_colors = np.concatenate([facecolors_before[:i], all_white[i:]])
            sc.set_facecolors(current_colors)
            sc.set_edgecolors(current_colors)
            return sc,

        facecolors_before = sc.get_facecolors()
        shape_colors = facecolors_before.shape
        all_white = np.zeros(shape_colors)

        self.ax.view_init(elev=10, azim=30)
        ani = FuncAnimation(self.fig, func=update, frames=len(facecolors_before), interval=5, repeat=False)
        writergif = PillowWriter(fps=3, bitrate=-1)
        # noinspection PyTypeChecker
        ani.save(f"{self.ani_path}{self.data_name}_order.gif", writer=writergif, dpi=400)

    def make_trans_animation(self, ax=None):
        dimension_index = -1
        points = self.sphere_grid.get_grid_as_array()
        # create the axis with the right num of dimensions
        self._create_fig_ax(ax, dim=points.shape[1]-1)
        self._set_up_empty()
        # sort by the value of the specific dimension you are looking at
        ind = np.argsort(points[:, dimension_index])
        points_3D = points[ind]
        # map the 4th dimension into values 0-1
        alphas = points_3D[:, dimension_index].T
        alphas = (alphas - np.min(alphas)) / np.ptp(alphas)

        all_points = []
        for line in points_3D:
            all_points.append(self.ax.scatter(*line[:self.dimensions], color="black", alpha=1))

        self._axis_limits(-1, 1, -1, 1, -1, 1)
        self._equalize_axes()

        step = 20

        def animate(frame):
            # plot current point
            current_time = alphas[frame * step]

            for i, p in enumerate(all_points):
                new_alpha = np.max([0, 1 - np.abs(alphas[i] - current_time) * 10])
                p.set_alpha(new_alpha)
            return self.ax,

        anim = FuncAnimation(self.fig, animate, frames=len(points_3D) // step,
                             interval=100)  # , frames=180, interval=50
        writergif = PillowWriter(fps=100 // step, bitrate=-1)
        # noinspection PyTypeChecker
        anim.save(f"{self.ani_path}{self.data_name}_trans.gif", writer=writergif, dpi=400)

    def create_all_plots(self, and_animations=False):
        self.make_grid_plot()
        self.make_alpha_plot()
        self.make_convergence_plot()
        self.make_grid_colored_with_alpha()
        if and_animations:
            self.make_rot_animation()
            self.make_ordering_animation()
            self.make_trans_animation()


class PolytopePlot(AbstractPlot):

    def __init__(self, polytope: Polytope, **kwargs):
        self.polytope = polytope
        split_name = str(polytope).split()
        data_name = f"{split_name[0]}_{split_name[-1]}"
        super().__init__(data_name, **kwargs)

    def make_graph(self, ax=None, with_labels=True):
        """
        Plot the networkx graph of self.G.
        """
        self._create_fig_ax(ax, dim=2)
        self._set_up_empty()
        node_labels = {i: tuple(np.round(i, 3)) for i in self.polytope.G.nodes}
        nx.draw_networkx(self.polytope.G, pos=nx.spring_layout(self.polytope.G, weight="p_dist"),
                         with_labels=with_labels, labels=node_labels)
        self.save_plot(name_addition="graph")

    def make_neighbours_plot(self, ax = None, node_i=0):
        """
        Want to see which points count as neighbours, second- or third neighbours of a specific node? Use this plotting
        method.
        """
        self._create_fig_ax(ax, dim=3, projection="3d")
        self._set_up_empty()
        if self.polytope.d != 3:
            print(f"Plotting neighbours not available for d={self.polytope.d}")
            return
        all_nodes = self.polytope.get_node_coordinates()
        node = tuple(all_nodes[node_i])
        neig = self.polytope.G.neighbors(node)
        sec_neig = list(second_neighbours(self.polytope.G, node))
        third_neig = list(third_neighbours(self.polytope.G, node))
        for sel_node in all_nodes:

            self.ax.scatter(*sel_node, color="black", s=30, alpha=0.5)
            if np.allclose(sel_node, node):
                self.ax.scatter(*sel_node, color="red", s=40, alpha=0.5)
            if tuple(sel_node) in neig:
                self.ax.scatter(*sel_node, color="blue", s=38, alpha=0.5)
            if tuple(sel_node) in sec_neig:
                self.ax.scatter(*sel_node, color="green", s=35, alpha=0.5)
            if tuple(sel_node) in third_neig:
                self.ax.scatter(*sel_node, color="orange", s=32, alpha=0.5)
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
