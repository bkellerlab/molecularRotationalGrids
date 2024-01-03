import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import geometric_slerp
from scipy.constants import pi

from molgri.plotting.abstract import RepresentationCollection, plot3D_method
from molgri.space.voronoi import AbstractVoronoi
from molgri.space.utils import normalise_vectors, sort_points_on_sphere_ccw, triangle_order_for_ccw_polygons
from molgri.wrappers import plot_method


class VoronoiPlot(RepresentationCollection):

    def __init__(self, voronoi: AbstractVoronoi, **kwargs):
        self.voronoi = voronoi
        N_points = len(self.voronoi.get_all_voronoi_centers())
        super().__init__(data_name=f"voronoi_{N_points}", **kwargs)

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
    def plot_borders(self, color="black", reduced=True):
        t_vals = np.linspace(0, 1, 2000)
        vertices = self.get_all_voronoi_vertices(reduced=reduced)
        regions = self.get_all_voronoi_regions(reduced=reduced)
        for i, region in enumerate(regions):
            n = len(region)
            for j in range(n):
                start = vertices[region][j]
                end = vertices[region][(j + 1) % n]
                norm = np.linalg.norm(start)
                # plot a spherical border
                if np.isclose(np.linalg.norm(start), np.linalg.norm(end)) and not np.isclose(np.linalg.norm(start), 0):
                    #print(np.linalg.norm(start), np.linalg.norm(end))
                    result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)
                    self.ax.plot(norm * result[..., 0], norm * result[..., 1], norm * result[..., 2], c=color)
                # plot a straight border
                else:
                    line = np.array([start, end])
                    self.ax.plot(*line.T, c=color)

        self._set_axis_limits()
        self._equalize_axes()

    @plot3D_method
    def plot_regions(self, color=None, reduced=True):
        #vertices = self.get_all_voronoi_vertices(reduced=reduced)
        regions = self.get_all_voronoi_regions(reduced=reduced)
        #additional = self._additional_points_per_cell()

        for i, region in enumerate(regions):
            self._plot_one_region(index_center=i, color=color, reduced=reduced)
            # # displays flat polygons
            # if not np.any(additional[i]):
            #     v = sort_points_on_sphere_ccw(vertices[region])
            #     #polygon = Poly3DCollection([v], alpha=0.5) #, facecolors=color
            #     #self.ax.add_collection3d(polygon)
            #     self.ax.plot_trisurf(*v.T, alpha=0.5)
            # # approximates rounded polygons
            # else:
            #     my_points = np.vstack([vertices[region], additional[i]])
            #     #self.ax.scatter(*my_points.T)
            #     my_points = sort_points_on_sphere_ccw(my_points)
            #     self.ax.plot_trisurf(*my_points.T, alpha=0.5) #color=color,

    def _plot_one_region(self, index_center: int, color=None, reduced=True, **kwargs):
        relevant_points = self.get_convex_hulls()[index_center].points
        triangles = self.get_convex_hulls()[index_center].simplices

        X, Y, Z = relevant_points.T

        self.ax.plot_trisurf(X, Y, triangles=triangles, alpha=0.5, color=color, linewidth=0, **kwargs, Z=Z)

    @plot3D_method
    def plot_vertices_of_i(self, index_center: int = 0, color="blue", labels=True, reduced=False, region=False):
        all_vertices = self.get_all_voronoi_vertices(reduced=reduced)
        all_regions = self.get_all_voronoi_regions(reduced=reduced)

        try:
            indices_of_i = all_regions[index_center]
        except IndexError:
            print(f"The grid does not contain index index_center={index_center}")
            return
        vertices_of_i = all_vertices[indices_of_i]

        self.ax.scatter(vertices_of_i[:, 0], vertices_of_i[:, 1], vertices_of_i[:, 2], c=color)

        if labels:
            for ic, point in enumerate(vertices_of_i):
                self.ax.text(*point[:3], s=f"{indices_of_i[ic]}", c=color)
        if region:
            self._plot_one_region(index_center=index_center, color=color, reduced=reduced)
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

    @plot_method
    def plot_center_distances_heatmap(self):
        my_array = self.get_center_distances().toarray()
        self._plot_position_N_N(my_array, cbar=False)


if __name__ == "__main__":
    from molgri.space.voronoi import RotobjVoronoi, PositionVoronoi
    from molgri.space.utils import normalise_vectors, random_sphere_points, random_quaternions
    import matplotlib.pyplot as plt
    np.random.seed(1)
    my_points = random_sphere_points(25)
    dists = np.array([0.1, 0.3])


    my_voronoi = RotobjVoronoi(my_points, using_detailed_grid=True)
    vp = VoronoiPlot(my_voronoi)
    vp.plot_centers(save=False)

    vp.plot_regions(ax=vp.ax, fig=vp.fig, save=False, animate_rot=False, )
    vp.plot_borders(ax=vp.ax, fig=vp.fig, save=True, animate_rot=True, reduced=True)


