"""
These plots are useful for presentations.
"""
import numpy as np
from scipy.constants import pi
from scipy.spatial import geometric_slerp, Voronoi, voronoi_plot_2d

from molgri.plotting.abstract import RepresentationCollection
from molgri.wrappers import plot3D_method, plot_method
from molgri.space.utils import normalise_vectors, random_sphere_points


class CurvedPlot(RepresentationCollection):

    def __init__(self):
        super().__init__(data_name="curved_plot", default_complexity_level="empty")

    @plot_method
    def plot_curve_vs_line(self):

        alpha = np.linspace(0, 2*pi, 8)
        xs, ys = np.cos(alpha), np.sin(alpha)
        # points
        self.ax.scatter(xs, ys, color="black")
        # straight line
        # self.ax.plot(xs[:2], ys[:2], color="red", label="Euclidean")
        # # curved line
        # t_vals = np.linspace(0, 1, 2000)
        # start = np.array([xs[0], ys[0]])
        # end = np.array([xs[1], ys[1]])
        # norm = np.linalg.norm(start)
        # result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)
        #
        # points = [[x, y] for x, y in zip(xs, ys)]
        # my_voronoi = Voronoi(points)
        # voronoi_plot_2d(my_voronoi, ax=self.ax)
        #self.ax.plot(norm * result[..., 0], norm * result[..., 1], c="blue", label="cosine")
        #self.ax.legend(loc="upper left")
        self._set_axis_limits()
        self._equalize_axes()


if __name__ == "__main__":
    cp = CurvedPlot()
    cp.plot_curve_vs_line()