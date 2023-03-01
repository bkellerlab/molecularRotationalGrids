import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from abc import ABC, abstractmethod

from molgri.space.fullgrid import FullGrid
from molgri.space.rotobj import SphereGridFactory
from molgri.space.utils import normalise_vectors
from molgri.plotting.fullgrid_plots import FullGridPlot
import matplotlib.pyplot as plt

# tests should always be performed on fresh data
USE_SAVED = False


class IdealPolyhedron(ABC):

    def __init__(self, Rs: NDArray, N_vertices):
        self.Rs = Rs
        self.N_vertices = N_vertices

    def get_ideal_sphere_area(self):
        """For each radius get ideal area of curved surfaces"""
        return 4 * pi * self.Rs ** 2 / self.N_vertices

    def get_ideal_sideways_area(self):
        """Get ideal areas separating points in the same level"""
        alpha = self.get_central_angle()
        circles = self.Rs**2 * alpha / 2
        areas = [circles[0]]
        for circ in circles[1:]:
            areas.append(circ - areas[-1])
        areas = np.array(areas)
        return areas

    @abstractmethod
    def get_central_angle(self):
        pass

    def get_ideal_volume(self):
        full_volumes = 4 / 3 * pi * self.Rs**3 / self.N_vertices
        volumes = [full_volumes[0]]
        for vol in full_volumes[1:]:
            volumes.append(vol - volumes[-1])
        volumes = np.array(volumes)
        return volumes


class IdealTetrahedron(IdealPolyhedron):

    def __init__(self, Rs):
        super().__init__(Rs, N_vertices=4)

    def get_central_angle(self):
        # tetrahedron is a dual of itself, so we here use the central angle of tetrahedron
        return np.arccos(-1/3)


class IdealIcosahedron(IdealPolyhedron):

    def __init__(self, Rs):
        super().__init__(Rs, N_vertices=12)

    def get_central_angle(self):
        # this is the central angle of a regular dodecahedron (vertex-origin-vertex). Why does that make sense?
        # The icosahedron and the dodecahedron are duals, so connecting the centers of the faces of an icosahedron
        # gives a dodecahedron and vice-versa. The shape of Voronoi cells on the sphere based on an icosahedron
        # partition is like a curved dodecahedron - has the same angles
        return np.arccos(np.sqrt(5)/3)


def _visualise_fg(fg7: FullGrid):
    fgp7 = FullGridPlot(fg7)
    fgp7.make_position_plot(numbered=True, save=False)
    fgp7.make_full_voronoi_plot(ax=fgp7.ax, fig=fgp7.fig, save=False, plot_vertex_points=True)
    plt.show()


def get_tetrahedron_grid(visualise=False, **kwargs):
    # simplest possible example: voronoi cells need at least 4 points to be created
    # with 4 points we expect tetrahedron angles
    fg = FullGrid(b_grid_name="zero", o_grid_name=f"cube3D_4", t_grid_name="[0.3]", **kwargs)
    fvg = fg.get_full_voronoi_grid()

    if visualise:
        _visualise_fg(fg)
    return fg, fvg


def get_icosahedron_grid(visualise=False, **kwargs):
    my_fg = FullGrid(b_grid_name="zero", o_grid_name=f"ico_12", t_grid_name="[0.2, 0.4]", **kwargs)
    print(my_fg.get_name())
    my_fvg = my_fg.get_full_voronoi_grid()

    if visualise:
        _visualise_fg(my_fg)
    return my_fg, my_fvg


def test_fullgrid_voronoi_radii():
    # we expect voronoi radii to be right in-between layers of points
    fg = FullGrid(b_grid_name="ico_7", o_grid_name=f"cube3D_15", t_grid_name="[0.3, 1, 2, 2.7, 3]", use_saved=False)
    # point radii as expected
    assert np.allclose(fg.get_radii(), [3, 10, 20, 27, 30])
    # between radii as expected; last one is same as previous distance
    assert np.allclose(fg.get_between_radii(), [6.5, 15, 23.5, 28.5, 31.5])
    # assert that those are also the radii of voronoi cells
    voronoi = fg.get_full_voronoi_grid()
    voronoi_radii = [sv.radius for sv in voronoi.get_voronoi_discretisation()]
    assert np.allclose(voronoi_radii, [6.5, 15, 23.5, 28.5, 31.5])

    # example with only one layer
    fg = FullGrid(b_grid_name="ico_7", o_grid_name=f"cube3D_15", t_grid_name="[0.3]")
    # point radii as expected
    assert np.allclose(fg.get_radii(), [3])
    # between radii as expected; last one is same as previous distance
    assert np.allclose(fg.get_between_radii(), [6])
    # assert that those are also the radii of voronoi cells
    voronoi = fg.get_full_voronoi_grid()
    voronoi_radii = [sv.radius for sv in voronoi.get_voronoi_discretisation()]
    assert np.allclose(voronoi_radii, [6])


def test_cell_assignment():
    # 1st test: the position grid points must correspond to themselves
    N_rot = 18
    fullgrid = FullGrid(b_grid_name="ico_7", o_grid_name=f"ico_{N_rot}", t_grid_name="[1, 2, 3]", use_saved=False)
    points_local = fullgrid.get_flat_position_grid()
    num_points = len(points_local)
    assert np.all(fullgrid.point2cell_position_grid(points_local) == np.arange(0, num_points))
    # 1st test:even if a bit of noise is added
    random_noise = 2 * (np.random.random(points_local.shape) - 0.5)
    points_local = points_local + random_noise
    assert np.all(fullgrid.point2cell_position_grid(points_local) == np.arange(0, num_points))
    # if you take a subset of points, you get the same result
    points_local = np.array([points_local[17], points_local[3], points_local[8], points_local[33]])
    assert np.all(fullgrid.point2cell_position_grid(points_local) == [17, 3, 8, 33])
    # points that are far out should return NaN
    points_local = np.array([[200, -12, 3], [576, -986, 38]])
    assert np.all(np.isnan(fullgrid.point2cell_position_grid(points_local)))
    between_radii = fullgrid.get_between_radii()
    # point with a radius < first voronoi radius must get an index < 35
    points_local = np.random.random((15, 3)) - 0.5
    points_local = normalise_vectors(points_local, length=between_radii[0]-1)
    assert np.all(fullgrid.point2cell_position_grid(points_local) < N_rot)
    # similarly for a second layer of radii
    points_local = np.random.random((15, 3)) - 0.5
    points_local = normalise_vectors(points_local, length=between_radii[0]+0.5)
    assert np.all(fullgrid.point2cell_position_grid(points_local) >= N_rot)
    assert np.all(fullgrid.point2cell_position_grid(points_local) < 2*N_rot)
    # and for third
    points_local = np.random.random((15, 3)) - 0.5
    points_local = normalise_vectors(points_local, length=between_radii[2] - 0.5)
    assert np.all(fullgrid.point2cell_position_grid(points_local) >= 2 * N_rot)
    assert np.all(fullgrid.point2cell_position_grid(points_local) < 3 * N_rot)


def test_division_area():
    fg, fvg = get_tetrahedron_grid(visualise=False, use_saved=False)

    # what we expect:
    # 1) all points are neighbours (in the same layer)
    # 2) all points share a division surface that approx equals R^2*alpha/2
    # where R is the voronoi radius and alpha the Vertex-Center-Vertex tetrahedron angle
    R = fg.get_between_radii()
    expected_surface = IdealTetrahedron(R).get_ideal_sideways_area()[0]

    all_div_areas = []
    for i in range(4):
        for j in range(i+1, 4):
            all_div_areas.append(fvg.get_division_area(i, j))
    # because all should be neighbours
    assert None not in all_div_areas
    all_div_areas = np.array(all_div_areas)
    # on average should be right area
    assert np.allclose(np.average(all_div_areas), expected_surface, rtol=0.01, atol=0.1)
    # for each individual, allowing for 5% error
    assert np.all(all_div_areas < 1.05 * expected_surface)
    assert np.all(all_div_areas > 0.95 * expected_surface)
    rel_errors = np.abs(all_div_areas - expected_surface) / expected_surface * 100
    print(f"Relative errors in tetrahedron surfaces: {np.round(rel_errors, 2)}")

    # the next example is with 12 points in form of an icosahedron and two layers

    fug2, fvg2 = get_icosahedron_grid(visualise=False, use_saved=False)

    R_s = fug2.get_between_radii()
    ii = IdealIcosahedron(R_s)
    areas_sideways = ii.get_ideal_sideways_area()
    areas_above = ii.get_ideal_sphere_area()

    # points 0 and 12, 1 and 13 ... right above each other
    real_areas_above = []
    for i in range(0, 12):
        real_areas_above.append(fvg2.get_division_area(i, i+12))
    real_areas_above = np.array(real_areas_above)
    assert np.allclose(real_areas_above, areas_above[0])

    real_areas_first_level = []
    # now let's see some sideways areas
    for i in range(12):
        for j in range(12):
            area = fvg2.get_division_area(i, j, print_message=False)
            if area is not None:
                real_areas_first_level.append(area)
    real_areas_first_level = np.array(real_areas_first_level)
    assert np.allclose(real_areas_first_level, areas_sideways[0])

    real_areas_sec_level = []
    # now let's see some sideways areas - this time second level of points
    for i in range(12, 24):
        for j in range(12, 24):
            area = fvg2.get_division_area(i, j, print_message=False)
            if area is not None:
                real_areas_sec_level.append(area)
    real_areas_sec_level = np.array(real_areas_sec_level)
    assert np.allclose(real_areas_sec_level, areas_sideways[1])


def test_volumes():
    # tetrahedron example
    fg, fvg = get_tetrahedron_grid(use_saved=False)
    real_vol = fvg.get_all_voronoi_volumes()
    R_s = fg.get_between_radii()

    it = IdealTetrahedron(R_s)
    ideal_vol = it.get_ideal_volume()
    assert np.isclose(np.average(real_vol), ideal_vol)

    # icosahedron example
    fg, fvg = get_icosahedron_grid(use_saved=False)
    real_vol = fvg.get_all_voronoi_volumes()
    R_s = fg.get_between_radii()

    ii = IdealIcosahedron(R_s)
    ideal_vol = ii.get_ideal_volume()
    assert np.allclose(real_vol[:12], ideal_vol[0])
    assert np.allclose(real_vol[12:], ideal_vol[1])


def test_default_full_grids():
    full_grid = FullGrid(t_grid_name="[1]", o_grid_name="zero", b_grid_name="zero")
    assert np.all(full_grid.t_grid.get_trans_grid() == np.array([10]))
    assert full_grid.get_position_grid().shape == (1, 1, 3)
    assert np.allclose(full_grid.get_position_grid(), np.array([[[0, 0, 10]]]))
    assert np.allclose(full_grid.b_rotations.rotations.as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[0]", o_grid_name="None", b_grid_name="None")
    assert np.all(full_grid.t_grid.get_trans_grid() == np.array([0]))
    assert full_grid.get_position_grid().shape == (1, 1, 3)
    assert np.allclose(full_grid.get_position_grid(), np.array([[[0, 0, 0]]]))
    assert np.allclose(full_grid.b_rotations.rotations.as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[1, 2, 3]", o_grid_name="0", b_grid_name="0")
    assert np.all(full_grid.t_grid.get_trans_grid() == np.array([10, 20, 30]))
    assert full_grid.get_position_grid().shape == (1, 3, 3)
    assert np.allclose(full_grid.get_position_grid(), np.array([[[0, 0, 10],
                                                                 [0, 0, 20],
                                                                 [0, 0, 30]]]))
    assert np.allclose(full_grid.b_rotations.rotations.as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[1, 2, 3]", o_grid_name="1", b_grid_name="1")
    assert np.all(full_grid.t_grid.get_trans_grid() == np.array([10, 20, 30]))
    assert full_grid.get_position_grid().shape == (1, 3, 3)
    assert np.allclose(full_grid.get_position_grid(), np.array([[[0, 0, 10],
                                                                 [0, 0, 20],
                                                                 [0, 0, 30]]]))
    assert np.allclose(full_grid.get_body_rotations().as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[1, 2, 3]", o_grid_name="3", b_grid_name="4")
    assert full_grid.get_position_grid().shape == (3, 3, 3)
    assert full_grid.b_rotations.get_grid_as_array().shape == (4, 4)


def test_position_grid():
    num_rot = 14
    num_trans = 4  # keep this number unless you change t_grid_name
    fg = FullGrid(b_grid_name="zero", o_grid_name=f"ico_{num_rot}", t_grid_name="[0.1, 2, 2.5, 4]")
    ico_ = SphereGridFactory.create(N=num_rot, alg_name="ico", dimensions=3, use_saved=USE_SAVED)
    ico_grid = ico_.get_grid_as_array()
    position_grid = fg.get_position_grid()
    assert position_grid.shape == (num_rot, num_trans, 3)
    # assert lengths correct throughout the array
    assert np.allclose(position_grid[:, 0], ico_grid)
    assert np.isclose(np.linalg.norm(position_grid[5][0]), 1)

    ico_grid2 = np.array([20*el for el in ico_grid])
    assert np.allclose(position_grid[:, 1], ico_grid2)
    assert np.isclose(np.linalg.norm(position_grid[-1][1]), 20)

    ico_grid3 = np.array([25*el for el in ico_grid])
    assert np.allclose(position_grid[:, 2], ico_grid3)
    assert np.isclose(np.linalg.norm(position_grid[3][2]), 25)

    ico_grid4 = np.array([40*el for el in ico_grid])
    assert np.allclose(position_grid[:, 3], ico_grid4)
    assert np.isclose(np.linalg.norm(position_grid[-1][3]), 40)

    # assert orientations stay the same
    for i in range(num_rot):
        selected_lines = position_grid[i, :]
        normalised_lines = normalise_vectors(selected_lines)
        assert np.allclose(normalised_lines, normalised_lines[0])


if __name__ == "__main__":
    #test_fullgrid_voronoi_radii()
    test_cell_assignment()
    test_division_area()
    test_volumes()
