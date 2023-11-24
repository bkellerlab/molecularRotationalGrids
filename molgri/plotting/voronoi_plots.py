import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import geometric_slerp
from scipy.constants import pi

from molgri.plotting.abstract import RepresentationCollection, plot3D_method
from molgri.space.voronoi import AbstractVoronoi
from molgri.space.utils import normalise_vectors
from molgri.wrappers import plot_method


class VoronoiPlot(RepresentationCollection):

    def __init__(self, voronoi: AbstractVoronoi):
        self.voronoi = voronoi
        N_points = len(self.voronoi.get_all_voronoi_centers())
        super().__init__(data_name=f"voronoi_{N_points}", default_complexity_level="half_empty")

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
    def plot_regions(self, color="green", reduced=True, additional_points=True):
        vertices = self.get_all_voronoi_vertices(reduced=reduced)
        regions = self.get_all_voronoi_regions(reduced=reduced)
        additional = self._additional_points_per_cell()
        for i, region in enumerate(regions):
            # displays flat polygons
            if not additional_points:
                polygon = Poly3DCollection([vertices[region]], alpha=0.5, facecolors=color)
                self.ax.add_collection3d(polygon)
            # approximates rounded polygons
            else:
                my_points = np.vstack([vertices[region], additional[i]])
                #self.ax.scatter(*my_points.T)
                self.ax.plot_trisurf(*my_points.T, alpha=0.5) #color=color,

    @plot3D_method
    def plot_vertices_of_i(self, i: int = 0, color="blue", labels=True, reduced=False, region=False):
        all_vertices = self.get_all_voronoi_vertices(reduced=reduced)
        all_regions = self.get_all_voronoi_regions(reduced=reduced)
        try:
            indices_of_i = all_regions[i]
        except IndexError:
            print(f"The grid does not contain index i={i}")
            return
        vertices_of_i = all_vertices[indices_of_i]
        self.ax.scatter(vertices_of_i[:, 0], vertices_of_i[:, 1], vertices_of_i[:, 2], c=color)

        if labels:
            for i, point in enumerate(vertices_of_i):
                self.ax.text(*point[:3], s=f"{indices_of_i[i]}", c=color)
        if region:
            polygon = Poly3DCollection([vertices_of_i], alpha=0.1, facecolors=color)
            self.ax.add_collection3d(polygon)
        self._set_axis_limits()
        self._equalize_axes()

    @plot3D_method
    def plot_volumes(self, approx=False):
        all_volumes = self.get_voronoi_volumes(approx=approx)
        self.plot_borders(ax=self.ax, fig=self.fig, save=False)
        self.plot_centers(ax=self.ax, fig=self.fig, labels=False, s=all_volumes, save=False)

    def _plot_position_N_N(self, my_array = None, **kwargs):
        sns.heatmap(my_array, cmap="gray", ax=self.ax, **kwargs)
        self._equalize_axes()

    @plot_method
    def plot_adjacency_heatmap(self):
        my_array = self.get_voronoi_adjacency().toarray()
        self._plot_position_N_N(my_array, cbar=False)

    @plot_method
    def plot_border_heatmap(self):
        my_array = self.get_cell_borders().toarray()
        self._plot_position_N_N(my_array, cbar=False)

if __name__ == "__main__":
    from molgri.space.voronoi import RotobjVoronoi, HalfRotobjVoronoi
    from molgri.space.utils import normalise_vectors, random_sphere_points, random_quaternions
    import matplotlib.pyplot as plt
    np.random.seed(1)
    my_points = random_quaternions(85)

    fig, ax = plt.subplots(1, 2)
    my_voronoi = RotobjVoronoi(my_points, using_detailed_grid=True)
    vp = VoronoiPlot(my_voronoi)
    vp.plot_border_heatmap(save=False, ax=ax[0], fig=fig)
    my_half_voronoi = my_voronoi.get_related_half_voronoi()
    vp_half = VoronoiPlot(my_half_voronoi)
    vp_half.plot_border_heatmap(save=False, ax=ax[1], fig=fig)
    plt.show()