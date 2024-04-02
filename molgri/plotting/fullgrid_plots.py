"""
Plots connected to the fullgrid module.

Plot position grids in space, Voronoi cells and their volumes etc.
"""

import numpy as np
import seaborn as sns
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.distance import cdist

from molgri.space.utils import normalise_vectors, random_sphere_points
from scipy.spatial import geometric_slerp

from molgri.constants import GRID_ALGORITHMS, NAME2SHORT_NAME
from molgri.plotting.abstract import RepresentationCollection, PanelRepresentationCollection, plot_voronoi_cells
from molgri.space.fullgrid import FullGrid, ConvergenceFullGridO


class FullGridPlot(RepresentationCollection):

    def __init__(self, full_grid: FullGrid, *args, **kwargs):
        self.full_grid = full_grid
        self.full_voronoi_grid = full_grid.get_full_voronoi_grid()
        data_name = self.full_grid.get_name()
        super().__init__(data_name, *args, **kwargs)

    def get_possible_title(self, algs = True, N_points = False):
        name = ""
        if algs:
            o_name = NAME2SHORT_NAME[self.full_grid.o_rotations.algorithm_name]
            b_name = NAME2SHORT_NAME[self.full_grid.b_rotations.algorithm_name]
            name += f"o = {o_name}, b = {b_name}"
        if N_points:
            N_o = self.full_grid.o_rotations.N
            N_b = self.full_grid.b_rotations.N
            N_t = self.full_grid.t_grid.get_N_trans()
            N_name = f"N_o = {N_o}, N_b = {N_b}, N_t = {N_t}"
            if len(name) > 0:
                N_name = f"; {N_name}"
            name += N_name
        return name

    def make_position_plot(self, ax=None, fig=None, save=True, animate_rot=False, numbered: bool = False,
                           c="black", projection="3d"):
        self._create_fig_ax(fig=fig, ax=ax, projection=projection)

        points = self.full_grid.get_flat_position_grid()
        self.ax.scatter(*points.T, color=c)

        if numbered:
            for i, point in enumerate(points):
                self.ax.text(*point, s=f"{i}")

        if projection == "3d":
            self.ax.view_init(elev=10, azim=30)
            self._set_axis_limits()
            self._equalize_axes()

        if save:
            self._save_plot_type(f"position_{projection}")
        if animate_rot:
            return self._animate_figure_view(self.fig, self.ax, f"position_rotated")

    def make_full_voronoi_plot(self, ax=None, fig=None, save=True, animate_rot=False, plot_vertex_points=True,
                               numbered: bool = False, colors=None, alphas=None):
        self._create_fig_ax(fig=fig, ax=ax, projection="3d")

        origin = np.zeros((3,))

        if numbered:
            points = self.full_grid.get_flat_position_grid()
            for i, point in enumerate(points):
                self.ax.text(*point, s=f"{i}", alpha=alphas[i])

        try:
            voronoi_disc = self.full_voronoi_grid.get_voronoi_discretisation()


            for i, sv in enumerate(voronoi_disc):
                sv.sort_vertices_of_regions()
                plot_voronoi_cells(sv, self.ax, plot_vertex_points=plot_vertex_points, colors=colors, alphas=alphas)
                # plot rays from origin to highest level
                if i == len(voronoi_disc)-1:
                    for vertex in sv.vertices:
                        ray_line = np.concatenate((origin[:, np.newaxis], vertex[:, np.newaxis]), axis=1)
                        self.ax.plot(*ray_line, color="black", alpha=alphas[i])
        except AttributeError:
            pass

        self.ax.view_init(elev=20, azim=20)
        self._set_axis_limits((-3, 3, -3, 3, -3, 3))
        self._equalize_axes()

        if save:
            self._save_plot_type("full_voronoi")
        if animate_rot:
            return self._animate_figure_view(self.fig, self.ax, f"full_voronoi_rotated")

    def make_point_vertices_plot(self, point_index: int, ax=None, fig=None, save=True, animate_rot=False,
                                 which="all"):
        self.make_full_voronoi_plot(ax=ax, fig=fig, save=False, plot_vertex_points=False)
        self.make_position_plot(save=False, numbered=True, ax=self.ax, fig=self.fig)

        try:
            vertices = self.full_voronoi_grid.find_voronoi_vertices_of_point(point_index, which=which)
            self.ax.scatter(*vertices.T, color="red")
        except AttributeError:
            pass

        save_name = f"vertices_of_{point_index}"
        if save:
            self._save_plot_type(save_name)
        if animate_rot:
            return self._animate_figure_view(self.fig, self.ax, save_name)

    def create_all_plots(self, and_animations=False):
        self.make_position_plot(numbered=True, animate_rot=and_animations)
        self.make_position_plot(numbered=False, animate_rot=and_animations)
        self.make_full_voronoi_plot(plot_vertex_points=True, animate_rot=and_animations)
        self.make_full_voronoi_plot(plot_vertex_points=False, animate_rot=and_animations)
        self.make_point_vertices_plot(0, animate_rot=and_animations)


class ConvergenceFullGridPlot(RepresentationCollection):

    def __init__(self, convergence_full_grid: ConvergenceFullGridO):
        self.convergence_full_grid = convergence_full_grid
        super().__init__(self.convergence_full_grid.get_name())

    def get_possible_title(self):
        full_name = self.convergence_full_grid.get_name()
        split_name = full_name.split("_")
        return NAME2SHORT_NAME[split_name[2]]

    def make_voronoi_volume_conv_plot(self, ax=None, fig=None, save=True):

        self._create_fig_ax(fig=fig, ax=ax)

        try:
            voronoi_df = self.convergence_full_grid.get_voronoi_volumes()

            all_layers = set(voronoi_df["layer"])

            for layer in all_layers:
                filtered_df = voronoi_df.loc[voronoi_df['layer'] == layer]
                sns.lineplot(data=filtered_df, x="N", y="Voronoi cell volume", errorbar="sd", ax=self.ax)
                sns.scatterplot(data=filtered_df, x="N", y="Voronoi cell volume", alpha=0.8, color="black", ax=self.ax,
                                s=1)
                sns.scatterplot(data=filtered_df, x="N", y="ideal volume", color="black", marker="x", ax=self.ax)
        except AttributeError:
            pass

        if save:
            self.ax.set_xscale("log")
            self._save_plot_type("voronoi_volume_conv")


class PanelConvergenceFullGridPlots(PanelRepresentationCollection):

    def __init__(self, t_grid_name: str, b_grid_name: str = "zero", N_set: list = None, **kwargs):
        list_plots = []
        for alg in GRID_ALGORITHMS[:-1]:
            conv_sphere_grid = ConvergenceFullGridO(o_alg_name=alg, N_set=N_set, b_grid_name=b_grid_name,
                                                    t_grid_name=t_grid_name, filter_non_unique=True, **kwargs)
            sphere_plot = ConvergenceFullGridPlot(conv_sphere_grid)
            list_plots.append(sphere_plot)
        data_name = f"all_convergence_{list_plots[0].convergence_full_grid.get_name()}"
        super().__init__(data_name, list_plots)

    def make_all_voronoi_volume_plots(self, save=True):
        self._make_plot_for_all("make_voronoi_volume_conv_plot")
        self.add_titles(list_titles=[subplot.get_possible_title() for subplot in self.list_plots])
        self.set_log_scale(x_axis=True, y_axis=False)
        self.unify_axis_limits()
        if save:
            self._save_multiplot("voronoi_volume")


if __name__ == "__main__":
    from molgri.constants import SMALL_NS, DEFAULT_NS, MINI_NS
    import matplotlib.pyplot as plt

    n_o = 42
    fg = FullGrid(f"zero", f"ico_{n_o}", "[0.15]", use_saved=False)
    colors = ["white"] * len(fg.get_flat_position_grid())
    vor_adj = fg.get_adjacency_of_position_grid().toarray()
    alphas = [0.0] * len(fg.get_flat_position_grid())
    #dist_adj = fg.get_poly_dist_adjacency()
    #poly_adj = fg.get_polyhedron_adjacency(o_grid=True).toarray()
    # cdis = cdist(fg.o_positions, fg.o_positions, "cos")
    # print(cdis)
    # wished = (8, 9, 10)

    # dist_neig = ((dist_adj[0] | poly_adj[0]) == 1)
    point_index = 5

    # for el in wished:
    #    colors[el] = "green"

    #for i, trug in enumerate(vor_adj[point_index]):
    #    if trug:
    #        colors[i] = "green"
    colors[point_index] = "blue"
    alphas[point_index] = 0.7

    fgp = FullGridPlot(fg, default_complexity_level="half_empty")
    #fgp.make_position_plot(save=False)
    #fgp.make_point_vertices_plot(point_index)
    ani = fgp.make_full_voronoi_plot(save=True, animate_rot=False, numbered=False, ax=fgp.ax, fig=fgp.fig,
                                     colors=colors, alphas=alphas)
    #plt.show()
    #ani = fgp.make_full_voronoi_plot(save=True, animate_rot=False, numbered=False, colors=colors)