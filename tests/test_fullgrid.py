import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from abc import ABC, abstractmethod

from molgri.space.fullgrid import FullGrid
from molgri.space.rotobj import SphereGridFactory
from molgri.space.utils import normalise_vectors, all_row_norms_equal_k
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
        alpha = self.get_dual_central_angle()
        circles = self.Rs**2 * alpha / 2
        areas = [circles[0]]
        for circ in circles[1:]:
            areas.append(circ - areas[-1])
        areas = np.array(areas)
        return areas

    @abstractmethod
    def get_dual_central_angle(self):
        #vertex-center-vertex angle
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

    def get_dual_central_angle(self):
        # tetrahedron is a dual of itself, so we here use the central angle of tetrahedron
        return np.arccos(-1/3)

    def get_vertex_center_vertex_angle(self):
        return np.arccos(-1 / 3)


class IdealIcosahedron(IdealPolyhedron):

    def __init__(self, Rs):
        super().__init__(Rs, N_vertices=12)

    def get_dual_central_angle(self):
        # this is the central angle of a regular dodecahedron (vertex-origin-vertex). Why does that make sense?
        # The icosahedron and the dodecahedron are duals, so connecting the centers of the faces of an icosahedron
        # gives a dodecahedron and vice-versa. The shape of Voronoi cells on the sphere based on an icosahedron
        # partition is like a curved dodecahedron - has the same angles
        return np.arccos(np.sqrt(5)/3)

    def get_vertex_center_vertex_angle(self):
        return 1.10714872


def _visualise_fg(fg7: FullGrid):
    fgp7 = FullGridPlot(fg7)
    fgp7.plot_positions(labels=True, save=False)
    fgp7.plot_position_voronoi(ax=fgp7.ax, fig=fgp7.fig, save=False, plot_vertex_points=True)
    plt.show()


def get_tetrahedron_grid(visualise=False, **kwargs):
    # simplest possible example: voronoi cells need at least 4 points to be created
    # with 4 points we expect tetrahedron angles
    fg = FullGrid(b_grid_name="randomQ_1", o_grid_name=f"cube3D_4", t_grid_name="[0.3]", **kwargs)

    if visualise:
        _visualise_fg(fg)
    return fg


def get_icosahedron_grid(visualise=False, **kwargs):
    my_fg = FullGrid(b_grid_name="1", o_grid_name=f"ico_12", t_grid_name="[0.2, 0.4]", **kwargs)

    if visualise:
        _visualise_fg(my_fg)
    return my_fg


def test_fullgrid_voronoi_radii():
    # we expect voronoi radii to be right in-between layers of points
    fg = FullGrid(b_grid_name="randomQ_7", o_grid_name=f"cube3D_15", t_grid_name="[0.3, 1, 2, 2.7, 3]", use_saved=False)
    # point radii as expected
    assert np.allclose(fg.get_radii(), [3, 10, 20, 27, 30])
    # between radii as expected; last one is same as previous distance
    assert np.allclose(fg.get_between_radii(), [6.5, 15, 23.5, 28.5, 31.5])
    # assert that those are also the radii of voronoi cells
    voronoi = fg.get_position_voronoi()
    voronoi_radii = [sv.radius for sv in voronoi]
    assert np.allclose(voronoi_radii, [6.5, 15, 23.5, 28.5, 31.5])

    # example with only one layer
    fg = FullGrid(b_grid_name="randomQ_7", o_grid_name=f"cube3D_15", t_grid_name="[0.3]")
    # point radii as expected
    assert np.allclose(fg.get_radii(), [3])
    # between radii as expected; last one is same as previous distance
    assert np.allclose(fg.get_between_radii(), [6])
    # assert that those are also the radii of voronoi cells
    voronoi = fg.get_position_voronoi()
    voronoi_radii = [sv.radius for sv in voronoi]
    assert np.allclose(voronoi_radii, [6])


def test_cell_assignment():
    # 1st test: the position grid points must correspond to themselves
    N_rot = 18
    fullgrid = FullGrid(b_grid_name="randomQ_7", o_grid_name=f"ico_{N_rot}", t_grid_name="[1, 2, 3]", use_saved=False)
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


def test_distances_voronoi_centers():
    # tetrahedron
    fg = get_tetrahedron_grid(visualise=False, use_saved=False)
    all_dist = fvg.get_all_distances_between_centers_as_numpy()
    rs = fg.get_radii()
    ideal_angle = IdealTetrahedron(rs).get_vertex_center_vertex_angle()
    ideal_dist = rs[0]*ideal_angle
    rel_errors = []
    for i in range(4):
        for j in range(i+1, 4):
            arch_dist = fvg.get_distance_between_centers(i, j, print_message=False)
            assert np.isclose(all_dist[i][j], arch_dist)
            assert np.isclose(all_dist[j][i], arch_dist)
            rel_errors.append(np.abs(arch_dist-ideal_dist)/ideal_dist * 100)
    print(f"Relative errors in tetrahedron distances {np.round(rel_errors, 2)}")

    # from molgri.plotting.other_plots import ArrayPlot
    # ArrayPlot(all_dist, data_name="tetra_distances").make_heatmap_plot(save=True)

    #icosahedron
    fg, fvg = get_icosahedron_grid(visualise=False, use_saved=False)
    rs = fg.get_radii()

    # all idealised distances
    ideal_ray_dist = rs[1] - rs[0]
    ideal_first_arch = rs[0] * IdealIcosahedron(rs).get_vertex_center_vertex_angle()
    ideal_second_arch = rs[1] * IdealIcosahedron(rs).get_vertex_center_vertex_angle()

    all_dist = fvg.get_all_distances_between_centers_as_numpy()

    # from molgri.plotting.other_plots import ArrayPlot
    # ArrayPlot(all_dist, data_name="ico_distances").make_heatmap_plot(save=True)

    # ray distances
    for i in range(12):
        assert np.isclose(all_dist[i, i+12], ideal_ray_dist)
        assert np.isclose(all_dist[i+12, i], ideal_ray_dist)

    # first circle distance
    for i in range(12):
        for j in range(i+1, 12):
            # if they are neighbours, they must have a specific distance
            if not np.isclose(all_dist[i, j], 0):
                assert np.isclose(all_dist[i, j], ideal_first_arch)
                assert np.isclose(all_dist[j, i], ideal_first_arch)

    # second circle distance
    for i in range(12, 24):
        for j in range(i + 1, 24):
            # if they are neighbours, they must have a specific distance
            if not np.isclose(all_dist[i, j], 0):
                assert np.isclose(all_dist[i, j], ideal_second_arch)
                assert np.isclose(all_dist[j, i], ideal_second_arch)


def test_division_area():
    fg, fvg = get_tetrahedron_grid(visualise=False, use_saved=False)
    # what we expect:
    # 1) all points are neighbours (in the same layer)
    # 2) all points share a division surface that approx equals R^2*alpha/2
    # where R is the voronoi radius and alpha the Vertex-Center-Vertex tetrahedron angle
    R = fg.get_between_radii()
    expected_surface = IdealTetrahedron(R).get_ideal_sideways_area()[0]

    # calculated all at once
    all_areas = fvg.get_all_voronoi_surfaces()
    all_areas = all_areas.toarray(order='C')

    all_div_areas = []
    for i in range(4):
        for j in range(i+1, 4):
            division_area = fvg.get_division_area(i, j)
            all_div_areas.append(division_area)
            assert np.isclose(division_area, all_areas[i, j])
            assert np.isclose(division_area, all_areas[j, i])
    # because all should be neighbours
    assert None not in all_div_areas
    all_div_areas = np.array(all_div_areas)
    # on average should be right area
    assert np.allclose(np.average(all_div_areas)/R, expected_surface/R, rtol=0.01, atol=0.1)
    rel_errors = np.abs(all_div_areas - expected_surface) / expected_surface * 100
    print(f"Relative errors in tetrahedron surfaces: {np.round(rel_errors, 2)}")

    # uncomment to visualise array
    #from molgri.plotting.other_plots import ArrayPlot
    #ArrayPlot(all_areas, data_name="tetra_areas").make_heatmap_plot(save=True)

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

    # uncomment to visualise array
    # all_areas = fvg2.get_all_voronoi_surfaces()
    # all_areas = all_areas.toarray(order='C')
    # from molgri.plotting.other_plots import ArrayPlot
    # ArrayPlot(all_areas, data_name="ico_areas").make_heatmap_plot(save=True)

    real_areas_sec_level = []
    # now let's see some sideways areas - this time second level of points
    for i in range(12, 24):
        for j in range(12, 24):
            area = fvg2.get_division_area(i, j, print_message=False)
            if area is not None:
                real_areas_sec_level.append(area)
    real_areas_sec_level = np.array(real_areas_sec_level)
    assert np.allclose(real_areas_sec_level, areas_sideways[1])

    # assert that curved areas add up to a full surface of sphere
    N_o = 22
    full_grid = FullGrid(t_grid_name="[3, 7]", o_grid_name=f"cube3D_{N_o}", b_grid_name="cube4D_6", use_saved=USE_SAVED)
    fvg = full_grid.get_full_voronoi_grid()
    first_radius = full_grid.get_between_radii()[0]
    exp_total_area = 4 * pi * first_radius ** 2
    areas = fvg.get_all_voronoi_surfaces_as_numpy()
    sum_curved_areas = 0
    for i in range(N_o):
        sum_curved_areas += areas[i, i+N_o]
    assert np.allclose(sum_curved_areas, exp_total_area)


def test_volumes():
    """
    This function tests that
    1) volumes are as expected for tetro- and icosahedron ideal examples
    2) for a random example the volumes add up to spheres and sphere cut-outs
    """
    # tetrahedron example
    # fg = get_tetrahedron_grid(use_saved=False)
    # real_vol = fg.get_all_position_volumes()
    # R_s = fg.get_between_radii()
    #
    # it = IdealTetrahedron(R_s)
    # ideal_vol = it.get_ideal_volume()
    # assert np.isclose(np.average(real_vol), ideal_vol)
    #
    # # icosahedron example
    # fg, fvg = get_icosahedron_grid(use_saved=False)
    # real_vol = fvg.get_all_voronoi_volumes()
    # R_s = fg.get_between_radii()
    #
    # ii = IdealIcosahedron(R_s)
    # ideal_vol = ii.get_ideal_volume()
    # assert np.allclose(real_vol[:12], ideal_vol[0])
    # assert np.allclose(real_vol[12:], ideal_vol[1])

    # test that volumes add up to a total volume of the largest sphere
    n_o = 56
    full_grid = FullGrid(t_grid_name="[3, 7]", o_grid_name=f"ico_{n_o}", b_grid_name="cube4D_6")
    r = full_grid.get_between_radii()**3
    #print(full_grid.get_o_grid().get_cell_volumes()[:7] * r[0])
    volumes = full_grid.get_all_position_volumes()
    #print(volumes[:7])
    print(full_grid.get_between_radii())
    print(np.sum(volumes[:n_o]), 4/3*pi*50**3)
    print(np.sum(volumes), 4 / 3 * pi * 90 ** 3)
    max_radius = full_grid.get_between_radii()[-1]
    exp_total_volume = 4/3 * pi * max_radius**3
    sum_volumes = np.sum(volumes)
    assert np.allclose(exp_total_volume, sum_volumes)


def test_default_full_grids():
    full_grid = FullGrid(t_grid_name="[1]", o_grid_name="1", b_grid_name="1", use_saved=False)
    assert np.all(full_grid.t_grid.get_trans_grid() == np.array([10]))
    assert full_grid.get_position_grid().shape == (1, 1, 3)
    assert np.allclose(full_grid.get_position_grid(), np.array([[[0, 0, 10]]]))
    assert np.allclose(full_grid.b_rotations.rotations.as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[0]", o_grid_name="None_1", b_grid_name="None_1", use_saved=False)
    assert np.all(full_grid.t_grid.get_trans_grid() == np.array([0]))
    assert full_grid.get_position_grid().shape == (1, 1, 3)
    assert np.allclose(full_grid.get_position_grid(), np.array([[[0, 0, 0]]]))
    assert np.allclose(full_grid.b_rotations.rotations.as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[1, 2, 3]", o_grid_name="ico_1", b_grid_name="cube4D_1", use_saved=False)
    assert np.all(full_grid.t_grid.get_trans_grid() == np.array([10, 20, 30]))
    assert full_grid.get_position_grid().shape == (1, 3, 3)
    assert np.allclose(full_grid.get_position_grid(), np.array([[[0, 0, 10],
                                                                 [0, 0, 20],
                                                                 [0, 0, 30]]]))
    assert np.allclose(full_grid.b_rotations.rotations.as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[1, 2, 3]", o_grid_name="1", b_grid_name="1", use_saved=False)
    assert np.all(full_grid.t_grid.get_trans_grid() == np.array([10, 20, 30]))
    assert full_grid.get_position_grid().shape == (1, 3, 3)
    assert np.allclose(full_grid.get_position_grid(), np.array([[[0, 0, 10],
                                                                 [0, 0, 20],
                                                                 [0, 0, 30]]]))
    assert np.allclose(full_grid.get_body_rotations().as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[1, 2, 3]", o_grid_name="3", b_grid_name="4", use_saved=False)
    assert full_grid.get_position_grid().shape == (3, 3, 3)
    assert full_grid.b_rotations.get_grid_as_array().shape == (4, 4)


def test_position_grid():
    """
    This function should test:
    1) position grid is created in the specified order, t.i. all orientations at first distance, then all orientations
    at second distance etc

    """
    num_rot = 14
    num_trans = 4  # keep this number unless you change t_grid_name
    t_grid = [0.1, 2, 2.5, 4]
    t_grid = [el*10 for el in t_grid]
    fg = FullGrid(b_grid_name="1", o_grid_name=f"ico_{num_rot}", t_grid_name=f"[0.1, 2, 2.5, 4]", use_saved=USE_SAVED)
    ico_ = SphereGridFactory.create(N=num_rot, alg_name="ico", dimensions=3, use_saved=USE_SAVED)
    ico_grid = ico_.get_grid_as_array()
    position_grid = fg.get_position_grid_as_array()

    assert position_grid.shape == (num_rot * num_trans, 3)
    # assert lengths of vectors correct throughout the array
    for i in range(num_trans):
        all_row_norms_equal_k(position_grid[i*num_rot:(i+1)*num_rot], t_grid[i])

    # assert orientations stay the same
    for i in range(num_trans):
        selected_lines = position_grid[i*num_rot:(i+1)*num_rot]
        normalised_lines = normalise_vectors(selected_lines)
        assert np.allclose(normalised_lines, ico_grid)



def test_voronoi_regression():
    """
    A collection of tests created by recording the current values of Voronoi areas, volumes or distances - when
    the implementation changes, these tests should confirm that the values are still correct.

    However, they are regression tests and should not be trusted unconditionally!
    """
    N_o = 22
    N_t = 2
    fg = FullGrid(b_grid_name="cube4D_16", o_grid_name=f"ico_{N_o}", t_grid_name="[2, 4]", use_saved=USE_SAVED)
    fvg = fg.get_position_voronoi()
    volumes = fvg.get_all_voronoi_volumes()
    expected_vols = np.array([5253.4525878,  8791.29124553, 8791.29124553, 7022.37191666, 7022.37191666,
                              5253.4525878,  4555.04327387, 4936.70923018])

    assert np.allclose(volumes[:8], expected_vols)

    all_areas = fvg.get_all_voronoi_surfaces()
    assert np.isclose(all_areas[13, 0], 0)
    assert np.isclose(all_areas[22, 0], 525.3452587797963)
    assert np.isclose(all_areas[11, 2], 0)
    assert np.isclose(all_areas[11, 4],  201.5729133338661)
    assert np.isclose(all_areas[41, 42], 0)

    # uncomment to visualise array
    # all_areas = all_areas.toarray(order='C')
    # from molgri.plotting.other_plots import ArrayPlot
    # ArrayPlot(all_areas).make_heatmap_plot(save=True)


def test_position_adjacency():
    # mini examples that you can calculate by hand
    fg = FullGrid(b_grid_name="randomQ_15", o_grid_name="ico_4", t_grid_name="[0.1, 0.2]", use_saved=USE_SAVED)
    # visualize
    # from molgri.plotting.fullgrid_plots import PositionGridPlot
    # import matplotlib.pyplot as plt
    # fgp = PositionGridPlot(fg)
    # fgp.plot_positions(save=False, labels=True)
    # fgp.plot_position_voronoi(ax=fgp.ax, fig=fgp.fig, save=False)
    # plt.show()

    n_points = 2*4
    # neighbours to everyone in same layer (not itself) + right above
    expected_adj = np.array([[False, True, True, True, True, False, False, False],
                             [True, False, True, True, False, True, False, False],
                             [True, True, False, True, False, False, True, False],
                             [True, True, True, False, False, False, False, True],
                             [True, False, False, False, False, True, True, True],
                             [False, True, False, False, True, False, True, True],
                             [False, False, True, False, True, True, False, True],
                             [False, False, False, True, True, True, True, False]])
    assert np.all(fg.get_adjacency_of_position_grid().toarray() == expected_adj)

    # examples where I know the answers
    # one radius
    fg = FullGrid("1", "ico_17", "[1,]", use_saved=USE_SAVED)
    my_array = fg.get_adjacency_of_position_grid().toarray()
    my_result = np.array([[False, False,  True, False, False, False,  True,  True, False,  True, False, False,
  False, False, False,  True, False],
 [False, False,  True, False, False,  True, False, False,  True,  True, False,  True,
   True, False,  True, False, False]])
    assert np.all(my_array[:2] == my_result)

    # two radii
    fg = FullGrid("1", "ico_10", "[1, 2]", use_saved=USE_SAVED)
    my_array = fg.get_adjacency_of_position_grid()
    adj_of_5 = [1, 2, 4, 6, 15]
    my_result = np.zeros(len(fg.get_flat_position_grid()), dtype=bool)
    my_result[adj_of_5] = True
    assert np.all(my_array.toarray()[5] == my_result)

    # three radii
    fg = FullGrid("cube4D_8", "ico_7", "linspace(1, 5, 3)", use_saved=USE_SAVED)
    # from molgri.plotting.fullgrid_plots import PositionGridPlot
    # import matplotlib.pyplot as plt
    # fgp = PositionGridPlot(fg)
    # fgp.plot_positions(save=False, labels=True)
    # fgp.plot_position_voronoi(ax=fgp.ax, fig=fgp.fig, save=False)
    # plt.show()
    my_array = fg.get_adjacency_of_position_grid().toarray()
    #_visualise_fg(fg)


    # bottom-layer
    adj_of_1 = [0, 2, 3, 5, 8]
    my_result = np.zeros(len(fg.get_flat_position_grid()), dtype=bool)
    my_result[adj_of_1] = True
    assert np.all(my_array[1] == my_result)

    # mid-layer
    adj_of_9 = [2, 7, 8, 12, 13, 16]
    my_result = np.zeros(len(fg.get_flat_position_grid()), dtype=bool)
    my_result[adj_of_9] = True
    assert np.all(my_array[9] == my_result)

    # top-layer
    adj_of_20 = [13, 14, 16, 18, 19]
    my_result = np.zeros(len(fg.get_flat_position_grid()), dtype=bool)
    my_result[adj_of_20] = True
    assert np.all(my_array[20] == my_result)

if __name__ == "__main__":
    test_fullgrid_voronoi_radii()
    #test_cell_assignment()
    #test_distances_voronoi_centers()
    #test_division_area()
    test_volumes()
    #test_default_full_grids()
    test_position_grid()
    #test_voronoi_regression()
    # test_position_adjacency()
