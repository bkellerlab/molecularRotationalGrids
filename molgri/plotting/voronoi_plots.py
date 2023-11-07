import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import geometric_slerp

from molgri.plotting.abstract import RepresentationCollection, plot3D_method
from molgri.space.voronoi import AbstractVoronoi
from molgri.wrappers import plot_method


class VoronoiPlot(RepresentationCollection):

    def __init__(self, voronoi: AbstractVoronoi):
        self.voronoi = voronoi
        super().__init__(data_name="voronoi", default_complexity_level="half_empty")

    def __getattr__(self, name):
        """ Enable forwarding methods to self.position_grid, so that from FullGrid you can access all properties and
         methods of PositionGrid too."""
        return getattr(self.voronoi, name)

    @plot3D_method
    def plot_centers(self, color="black", labels=True, **kwargs):
        points = self.get_all_voronoi_centers()
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, **kwargs)

        if labels:
            for i, point in enumerate(points):
                self.ax.text(*point[:3], s=f"{i}", c=color)

        self._set_axis_limits()
        self._equalize_axes()

    @plot3D_method
    def plot_vertices(self, color="green", labels=True, reduced=False, **kwargs):
        vertices = self.get_all_voronoi_vertices(reduced=reduced)

        self.ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=color, **kwargs)

        if labels:
            for i, line in enumerate(vertices):
                self.ax.text(*line[:3], i, c=color)

        self._set_axis_limits()
        self._equalize_axes()

    @plot3D_method
    def plot_borders(self, color="black", reduced=False):
        t_vals = np.linspace(0, 1, 2000)
        vertices = self.get_all_voronoi_vertices(reduced=reduced)
        regions = self.get_all_voronoi_regions(reduced=reduced)
        for i, region in enumerate(regions):
            n = len(region)
            for j in range(n):
                start = vertices[region][j]
                end = vertices[region][(j + 1) % n]
                norm = np.linalg.norm(start)
                result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)
                self.ax.plot(norm * result[..., 0], norm * result[..., 1], norm * result[..., 2], c=color)

        self._set_axis_limits()
        self._equalize_axes()

    @plot3D_method
    def plot_regions(self, color="green", reduced=True):
        vertices = self.get_all_voronoi_vertices(reduced=reduced)
        regions = self.get_all_voronoi_regions(reduced=reduced)
        for i, region in enumerate(regions):
            n = len(region)
            for j in range(n):
                polygon = Poly3DCollection([vertices[region]], alpha=0.1, facecolors=color)
                self.ax.add_collection3d(polygon)

    @plot3D_method
    def plot_vertices_of_i(self, i: int = 0, color="blue", labels=True, reduced=False):
        all_vertices = self.get_all_voronoi_vertices(reduced=reduced)
        all_regions = self.get_all_voronoi_regions(reduced=reduced)
        indices_of_i = all_regions[i]
        vertices_of_i = all_vertices[indices_of_i]
        self.ax.scatter(vertices_of_i[:, 0], vertices_of_i[:, 1], vertices_of_i[:, 2], c=color)

        if labels:
            for i, point in enumerate(vertices_of_i):
                self.ax.text(*point[:3], s=f"{indices_of_i[i]}", c=color)

        self._set_axis_limits()
        self._equalize_axes()


    @plot_method
    def plot_position_volumes(self):
        all_volumes = self.get_all_position_volumes()
        self.ax.scatter(all_volumes)

    def _plot_position_N_N(self, my_array = None, **kwargs):
        sns.heatmap(my_array, cmap="gray", ax=self.ax, **kwargs)
        self._equalize_axes()

    @plot_method
    def plot_position_adjacency(self):
        my_array = self.get_adjacency_of_position_grid().toarray()
        self._plot_position_N_N(my_array, cbar=False)

if __name__ == "__main__":
    from molgri.space.voronoi import RotobjVoronoi, HalfRotobjVoronoi
    from molgri.space.utils import normalise_vectors, random_sphere_points, random_quaternions
    import matplotlib.pyplot as plt
    np.random.seed(1)
    my_points = random_sphere_points(45)
    my_voronoi = HalfRotobjVoronoi(my_points)
    vp = VoronoiPlot(my_voronoi)
    vp.plot_centers(save=False)
    vp.plot_vertices(ax=vp.ax, fig=vp.fig, save=False, alpha=0.5, labels=False)
    vp.plot_borders(ax=vp.ax, fig=vp.fig, save=False)
    vp.plot_vertices_of_i(i=1, ax=vp.ax, fig=vp.fig, save=False)
    #vp.plot_regions(ax=vp.ax, fig=vp.fig, save=False)
    plt.show()