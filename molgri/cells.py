"""
This module builds cells that enable space discretisation into cells based on points defined by the Grids.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import SphericalVoronoi

from molgri.utils import norm_per_axis, normalise_vectors


def voranoi_surfaces_on_sphere(points: NDArray):
    """

    Args:
        points: (N, 3)-dimensional array in which each line is a point on a surface of a sphere

    Returns:

    """
    assert points.shape[1] == 3, "Must provide an input array of size (N, 3)!"
    norms = norm_per_axis(points)
    assert np.allclose(norms, norm_per_axis(points[0])), "All input points must have the same norm."
    return SphericalVoronoi(points, radius=norms[0, 0], threshold=1e-4)


if __name__ == "__main__":
    from molgri.grids import FullGrid
    import matplotlib.pyplot as plt
    from scipy.spatial import geometric_slerp
    fg = FullGrid(b_grid_name="zero", o_grid_name=f"systemE_32", t_grid_name="[2]")
    my_points = fg.get_position_grid()
    sv = voranoi_surfaces_on_sphere(my_points)
    # sort vertices (optional, helpful for plotting)

    sv.sort_vertices_of_regions()

    t_vals = np.linspace(0, 1, 2000)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # plot the unit sphere for reference (optional)

    u = np.linspace(0, 2 * np.pi, 100)

    v = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(u), np.sin(v))

    y = np.outer(np.sin(u), np.sin(v))

    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='y', alpha=0.1)

    # plot generator points

    ax.scatter(my_points[:, 0], my_points[:, 1], my_points[:, 2], c='b')

    # plot Voronoi vertices

    ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],

               c='g')

    # indicate Voronoi regions (as Euclidean polygons)

    for region in sv.regions:

        n = len(region)

        for i in range(n):
            start = sv.vertices[region][i]

            end = sv.vertices[region][(i + 1) % n]

            norm = np.linalg.norm(start)
            result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)

            ax.plot(norm*result[..., 0],

                    norm*result[..., 1],

                    norm*result[..., 2],

                    c='k')

    ax.azim = 10

    ax.elev = 40

    _ = ax.set_xticks([])

    _ = ax.set_yticks([])

    _ = ax.set_zticks([])

    fig.set_size_inches(4, 4)

    plt.show()