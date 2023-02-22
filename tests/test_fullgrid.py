import numpy as np

from molgri.space.fullgrid import FullGrid
from molgri.space.utils import normalise_vectors


def test_cell_assignment():
    # 1st test: the position grid points must correspond to themselves
    N_rot = 35
    fg = FullGrid(b_grid_name="ico_7", o_grid_name=f"ico_{N_rot}", t_grid_name="[1, 2, 3]")
    points = fg.get_flat_position_grid()
    num_points = len(points)
    assert np.all(fg.point2cell_position_grid(points) == np.arange(0, num_points))
    # 1st test:even if a bit of noise is added
    random_noise = 2 * (np.random.random(points.shape) - 0.5)
    points = points + random_noise
    assert np.all(fg.point2cell_position_grid(points) == np.arange(0, num_points))
    # if you take a subset of points, you get the same result
    points = np.array([points[17], points[3], points[8], points[33]])
    assert np.all(fg.point2cell_position_grid(points) == [17, 3, 8, 33])
    # points that are far out should return NaN
    points = np.array([[200, -12, 3], [576, -986, 38]])
    assert np.all(np.isnan(fg.point2cell_position_grid(points)))
    between_radii = fg.get_between_radii()
    # point with a radius < first voronoi radius must get an index < 35
    points = np.random.random((15, 3)) - 0.5
    points = normalise_vectors(points, length=between_radii[0]-1)
    assert np.all(fg.point2cell_position_grid(points) < N_rot)
    # similarly for a second layer of radii
    points = np.random.random((15, 3)) - 0.5
    points = normalise_vectors(points, length=between_radii[0]+0.5)
    assert np.all(fg.point2cell_position_grid(points) >= N_rot)
    assert np.all(fg.point2cell_position_grid(points) < 2*N_rot)
    # and for third
    points = np.random.random((15, 3)) - 0.5
    points = normalise_vectors(points, length=between_radii[2] - 0.5)
    assert np.all(fg.point2cell_position_grid(points) >= 2 * N_rot)
    assert np.all(fg.point2cell_position_grid(points) < 3 * N_rot)


def test_division_area():
    pass

if __name__ == "__main__":
    test_cell_assignment()