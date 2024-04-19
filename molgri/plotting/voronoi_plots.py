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
        regions = self.get_all_voronoi_regions(reduced=reduced)

        for i, region in enumerate(regions):
            self.plot_one_region(index_center=i, color=color)

    @plot3D_method
    def plot_one_region(self, index_center: int, color=None, alpha=0.5):
        relevant_points = self.get_convex_hulls()[index_center].points
        #self.ax.scatter(*relevant_points.T, s=2)
        triangles = self.get_convex_hulls()[index_center].simplices

        # don't move this to one line since it may have 4 components (hyperspheres)
        X = relevant_points.T[0]
        Y = relevant_points.T[1]
        Z = relevant_points.T[2]

        self.ax.plot_trisurf(X, Y, triangles=triangles, alpha=alpha, color=color, linewidth=0, Z=Z)

    @plot3D_method
    def plot_vertices_of_i(self, index_center: int = 0, color="blue", labels=True, reduced=False, region=False,
                           alpha=0.5):
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
            self.plot_one_region(index_center=index_center, color=color, alpha=alpha)
        self._set_axis_limits()
        self._equalize_axes()

    @plot_method
    def plot_volumes(self, approx=False):
        all_volumes = self.get_voronoi_volumes(approx=approx)
        sns.catplot(all_volumes, ax=self.ax, kind="violin")

    def _plot_position_N_N(self, my_array = None, **kwargs):
        sns.heatmap(my_array, cmap="gray", ax=self.ax, xticklabels=False, yticklabels=False, **kwargs)
        #self.ax.xaxis.tick_top()
        self._equalize_axes()

    @plot_method
    def plot_adjacency_heatmap(self):
        my_array = self.get_voronoi_adjacency().toarray()
        self._plot_position_N_N(my_array, cbar=False)

    @plot_method
    def plot_border_heatmap(self):
        my_array = self.get_cell_borders().toarray()
        print(my_array/(2*pi)*360)
        self._plot_position_N_N(my_array, cbar=True)

    @plot_method
    def plot_center_distances_heatmap(self):
        my_array = self.get_center_distances().toarray()
        self._plot_position_N_N(my_array, cbar=True)


if __name__ == "__main__":
    from molgri.space.rotobj import SphereGrid3DFactory
    from molgri.space.fullgrid import PositionGrid
    from molgri.space.voronoi import RotobjVoronoi, HalfRotobjVoronoi
    from molgri.space.utils import normalise_vectors, random_sphere_points, random_quaternions
    import matplotlib.pyplot as plt
    np.random.seed(1)
    my_points = SphereGrid3DFactory.create(alg_name="ico", N=42)

    voronoi1 = my_points.get_spherical_voronoi()
    voronoi2 = HalfRotobjVoronoi(normalise_vectors(my_points.get_grid_as_array(), length=1.5))

    vp = VoronoiPlot(voronoi1, default_complexity_level="empty")
    vp2 = VoronoiPlot(voronoi2, default_complexity_level="empty")
    vp.plot_adjacency_heatmap()
    vp.plot_border_heatmap()

    my_point = 7
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.view_init(0, 0)
    for i, length in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1]):
        voronoi2 = RotobjVoronoi(normalise_vectors(my_points.get_grid_as_array(), length=length))
        vp2 = VoronoiPlot(voronoi2, default_complexity_level="empty")
        if i==6:
            save=True
        else:
            save=False
        vp2.plot_centers(save=False, labels=False, fig=fig, ax=ax, s=35)

    #vp.plot_one_region(my_point, "blue", save=False, fig=vp.fig, ax=vp.ax)
    #vp.plot_borders(save=False, fig=vp.fig, ax=vp.ax)

    #vp2.plot_centers(save=False, labels=False, fig=vp.fig, ax=vp.ax)
    #vp2.plot_borders(save=False, fig=vp.fig, ax=vp.ax)
    #vp2.plot_one_region(my_point, "blue", save=False, fig=vp.fig, ax=vp.ax)


    # plot sides
    # coo1 = voronoi1.get_all_voronoi_vertices()[voronoi1.get_all_voronoi_regions()[my_point]]
    # coo1 = sort_points_on_sphere_ccw(coo1)
    #
    # coo2 = voronoi2.get_all_voronoi_vertices()[voronoi2.get_all_voronoi_regions()[my_point]]
    # coo2 = sort_points_on_sphere_ccw(coo2)
    # print("sorted twice")
    #
    # x = [coo1[0][0], coo1[1][0], coo2[1][0], coo2[0][0]]
    # y = [coo1[0][1], coo1[1][1], coo2[1][1], coo2[0][1]]
    # z = [coo1[0][2], coo1[1][2], coo2[1][2], coo2[0][2]]
    # verts = [list(zip(x, y, z))]
    # my_coll = Poly3DCollection(verts)
    # my_coll.set_color("blue")
    # #vp.ax.add_collection3d(my_coll)
    #
    #
    # mylim = 1.5
    # #vp.ax.set_xlim(-mylim, mylim)
    # #vp.ax.set_ylim(-mylim, mylim)
    # #vp.ax.set_zlim(-mylim, mylim)
    plt.show()





