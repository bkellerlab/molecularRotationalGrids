import numpy as np
from scipy.constants import pi

from molgri.space.fullgrid import FullGrid
from molgri.space.rotobj import SphereGridFactory
from molgri.space.utils import normalise_vectors

# tests should always be performed on fresh data
USE_SAVED = False


def test_fullgrid_voronoi_radii():
    # we expect voronoi radii to be right in-between layers of points
    fg = FullGrid(b_grid_name="ico_7", o_grid_name=f"cube3D_15", t_grid_name="[0.3, 1, 2, 2.7, 3]")
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
    # simplest possible example: voronoi cells need at least 4 points to be created
    # with 4 points we expect tetrahedron angles
    fg = FullGrid(b_grid_name="zero", o_grid_name=f"cube3D_4", t_grid_name="[3]")
    fvg = fg.get_full_voronoi_grid()

    # to visualise, uncomment
    # from molgri.plotting.fullgrid_plots import FullGridPlot
    # import matplotlib.pyplot as plt
    # fgp = FullGridPlot(fg)
    # fgp.make_position_plot(numbered=True, save=False)
    # fgp.make_full_voronoi_plot(ax=fgp.ax, fig=fgp.fig, save=False, plot_vertex_points=True)
    # plt.show()

    # what we expect:
    # 1) all points are neighbours (in the same layer)
    # 2) all points share a division surface that approx equals R^2*alpha/2
    # where R is the voronoi radius and alpha the Vertex-Center-Vertex tetrahedron angle
    R = fg.get_between_radii()[0]
    alpha = np.arccos(-1/3)
    expected_surface = R**2 * alpha / 2
    all_div_areas = []
    for i in range(4):
        for j in range(i+1, 4):
            all_div_areas.append(fvg.get_division_area(i, j))
    # because all should be neighbours
    assert None not in all_div_areas
    # allowing for 5% error
    assert np.all(all_div_areas < 1.05 * expected_surface)
    assert np.all(all_div_areas > 0.95 * expected_surface)

    # the next example is with 12 points in form of an icosahedron and two layers
    fg = FullGrid(b_grid_name="zero", o_grid_name=f"ico_12", t_grid_name="[2, 4]")
    fvg = fg.get_full_voronoi_grid()

    # to visualise, uncomment
    from molgri.plotting.fullgrid_plots import FullGridPlot
    import matplotlib.pyplot as plt
    fgp = FullGridPlot(fg)
    fgp.make_position_plot(numbered=True, save=False)
    fgp.make_full_voronoi_plot(ax=fgp.ax, fig=fgp.fig, save=False, plot_vertex_points=True)
    plt.show()

    R_s = fg.get_between_radii()
    alpha = np.arccos(-np.sqrt(5)/3)
    areas_sideways = R_s ** 2 * alpha / 2
    areas_above = 4 * pi * R_s ** 2 / 12

    # points 0 and 12, 1 and 13 ... right above each other
    real_areas_above = []
    for i in range(0, 12):
        real_areas_above.append(fvg.get_division_area(i, i+12))
    real_areas_above = np.array(real_areas_above)
    # on average, the areas are perfect
    assert np.allclose(np.average(real_areas_above), areas_above[0])
    # on their own, they are less impressive - see the relative errors
    rel_errors = np.abs(real_areas_above - areas_above[0])/areas_above[0] * 100
    print(f"Relative errors in icosahedron surfaces above first layer: {rel_errors}")

    real_areas_first_level = []
    # now let's see some sideways areas
    for i in range(12):
        for j in range(12):
            area = fvg.get_division_area(i, j)
            if area is not None:
                real_areas_first_level.append(area)
    real_areas_first_level = np.array(real_areas_first_level)
    print(real_areas_first_level)
    rel_errors = np.abs(real_areas_first_level - areas_sideways[0]) / areas_sideways[0] * 100
    print(f"Relative errors in icosahedron surfaces sideways in the first layer: {rel_errors}")




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
    #test_cell_assignment()
    test_division_area()
