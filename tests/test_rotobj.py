import pandas as pd
import numpy as np
from scipy.constants import pi
from scipy.linalg import issymmetric

from molgri.space.rotobj import SphereGrid3DFactory, SphereGrid4DFactory, SphereGridFactory
from molgri.constants import GRID_ALGORITHMS_3D, GRID_ALGORITHMS_4D, DEFAULT_ALPHAS_3D
from molgri.space.utils import all_row_norms_equal_k, two_sets_of_quaternions_equal, which_row_is_k

# tests should always be performed on fresh data
USE_SAVED = False

# don't test fulldiv
GRID_ALGORITHMS_4D = [x for x in GRID_ALGORITHMS_4D if x!="fulldiv"]

def test_saving_rotobj():
    """
    This function tests:
    1) after saving you get back the same grid (and therefore same statistics)
    """
    for algos, d in zip((GRID_ALGORITHMS_3D, GRID_ALGORITHMS_4D), (3, 4)):
        for N in (12, 23, 51):
            for algo in algos:
                rotobj_start = SphereGridFactory.create(N=N, alg_name=algo, dimensions=d, use_saved=False)
                array_before = rotobj_start.get_grid_as_array()
                statistics_before = rotobj_start.get_uniformity_df(alphas=DEFAULT_ALPHAS_3D)
                rotobj_new = SphereGridFactory.create(N=N, alg_name=algo, dimensions=d, use_saved=True)
                array_after = rotobj_new.get_grid_as_array()
                statistics_after = rotobj_new.get_uniformity_df(alphas=DEFAULT_ALPHAS_3D)
                assert rotobj_new.get_name(with_dim=True) == rotobj_start.get_name(with_dim=True)
                assert statistics_before.equals(statistics_after)
                assert np.allclose(array_before, array_after)


def test_general_grid_properties():
    """
    This function tests:
    1) each algorithm (except zero and fulldiv) is able to create grids in 3D and 4D with different num of points
    2) the type and size of the grid is correct
    3) grid points are normed to length 1
    """
    for algos, d in zip((GRID_ALGORITHMS_3D, GRID_ALGORITHMS_4D), (3, 4)):
        for number in (3, 15, 26):
            for alg in algos:
                grid_obj = SphereGridFactory.create(N=number, alg_name=alg, dimensions=d, use_saved=USE_SAVED)
                grid = grid_obj.get_grid_as_array()
                assert isinstance(grid, np.ndarray), "Grid must be a numpy array."
                assert grid.shape == (number, d), f"Wrong grid shape {grid.shape} for alg={alg}, N={number}, d={d}."
                assert grid_obj.N == number
                all_row_norms_equal_k(grid, 1)


def test_statistics():
    num_points = 35
    num_random = 50
    # grid statistics
    icog = SphereGridFactory.create(N=num_points, alg_name="ico", dimensions=3, use_saved=False)
    default_alphas = [pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6]
    icog.save_uniformity_statistics(num_random=num_random, alphas=default_alphas)
    statistics_csv = pd.read_csv(icog.get_statistics_path("csv"), index_col=0, header=0, dtype=float)
    assert len(statistics_csv) == len(default_alphas)*num_random
    expected_coverages = [0.0669872981077806, 0.2499999999999999,
                          0.4999999999999999, 0.7499999999999999, 0.9330127018922194]
    ideal_coverage = statistics_csv["ideal coverage"].to_numpy(dtype=float).flatten()
    for i, _ in enumerate(default_alphas):
        written_id_coverage = ideal_coverage[i*num_random:(i+1)*num_random-1]
        assert np.allclose(written_id_coverage, expected_coverages[i])


def test_ordering():
    """
    This function tests for ico/cube3D in 3D and for cube4D in 4D:
    1) if you create an object with N elements once and with N+M elements next time, the first N elements will be
    identical
    """
    for name in ["ico", "cube3D"]:
        for N in range(14, 111, 7):
            for addition in (1, 20, 3):
                grid_1 = SphereGridFactory.create(N=N+addition, alg_name=name, dimensions=3, use_saved=USE_SAVED)
                grid_1 = grid_1.get_grid_as_array()
                grid_2 = SphereGridFactory.create(N=N, alg_name=name, dimensions=3, use_saved=USE_SAVED)
                grid_2 = grid_2.get_grid_as_array()
                assert np.allclose(grid_1[:N], grid_2)
    for N in range(7, 111, 25):
        for addition in (1, 25, 4):
            grid_1 = SphereGridFactory.create(N=N + addition, alg_name="cube4D", dimensions=4,
                                              use_saved=USE_SAVED)
            grid_1 = grid_1.get_grid_as_array()
            grid_2 = SphereGridFactory.create(N=N, alg_name="cube4D", dimensions=4, use_saved=USE_SAVED)
            grid_2 = grid_2.get_grid_as_array()
            assert np.allclose(grid_1[:N], grid_2)


def test_voronoi_exact_divisions():
    """
    This function tests that:
    1) Voronoi areas can be calculated for all algorithms (except zero)
    2) for cube3D and ico, approximate areas are close to exact areas
    3) for cube3D, ico and cube4D, approximate areas are close to the theoretical prediction obtained by
    systematically dissecting a (hyper)sphere.
    """
    sphere_surface = 4*pi
    hypersphere_surface = pi**2   # only for half-hypersphere

    for name in GRID_ALGORITHMS_3D:
        try:
            my_grid = SphereGridFactory.create(N=45, alg_name=name, dimensions=3, use_saved=USE_SAVED)
            my_areas = my_grid.get_cell_volumes()
            # Voronoi areas can be calculated for all algorithms, they have the right length, are + and add up to 4pi
            assert len(my_areas) == 45
            assert np.all(my_areas > 0)
            assert np.isclose(np.sum(my_areas), sphere_surface)
            # compare border len and dist between points with polytope side len

        except ValueError:
            print(f"Duplicate generator issue for {name}.")


    # exact divisions of cube3D: 8, 26, 98
    for i, N in enumerate([8, 26, 98]):
        my_grid = SphereGrid3DFactory.create("cube3D", N, use_saved=False)
        # if you wanna visualize it, uncomment
        # from molgri.plotting.spheregrid_plots import SphereGridPlot
        # import matplotlib.pyplot as plt
        # sg = SphereGridPlot(my_grid)
        # sg.plot_grid(save=False, labels=True)
        # sg.plot_voronoi(ax=sg.ax, fig=sg.fig, save=True, labels=False, animate_rot=True)
        # plt.show()

        # areas
        # each surface individually should be somewhat close to theoretical
        theo_area = sphere_surface / N
        exact_areas = my_grid.get_cell_volumes(approx=False)
        approx_areas = my_grid.get_cell_volumes(approx=True, using_detailed_grid=True)
        # approximate areas are close to exact areas
        assert np.allclose(exact_areas, approx_areas, atol=1e-2, rtol=0.1)
        assert np.isclose(np.average(approx_areas), theo_area, atol=0.01, rtol=0.1)
        # sum of approximate areas also close to theoretical
        assert np.isclose(np.sum(approx_areas), sphere_surface, atol=0.1, rtol=0.1)
        # even if not using detailed grid, the values should be somewhat close
        not_detailed = my_grid.get_cell_volumes(approx=True, using_detailed_grid=False)
        # all are reasonably close to each other
        assert np.std(approx_areas) < 0.08
        assert np.std(not_detailed) < 0.08
        # no super weird values
        assert np.all(approx_areas < 2)
        assert np.all(approx_areas > 0.05)
        assert np.all(not_detailed < 2)
        assert np.all(not_detailed > 0.05)


        # lengths between neighbouring points
        # real len will be a bit longer than side_len because they are curved
        nonzero_dist = my_grid.get_center_distances().data
        side_len = 2 * np.sqrt(1/3) /(2**i)
        curvature_factor = 1.066
        real_dist, real_count = np.unique(np.round(nonzero_dist, 3), return_counts=True)
        if i == 0:
            assert np.isclose(side_len*curvature_factor, real_dist[0], atol=0.001)
        if i == 1:
            # one big circle of 8 elements in the middle
            array_dist = my_grid.get_center_distances().toarray()
            big_circle_el = [array_dist[17][25], array_dist[25][22], array_dist[22][8], array_dist[8][11],
                             array_dist[11][23], array_dist[23][15], array_dist[15][19], array_dist[19][17]]
            assert np.allclose(big_circle_el, 2*pi/8)
            # straight and diagonal elements
            straight_19_neig = [array_dist[19][17], array_dist[19][9], array_dist[19][15], array_dist[19][14]]
            diag_19_neig = [array_dist[19][5], array_dist[19][0], array_dist[19][3], array_dist[19][4]]
            assert np.allclose(straight_19_neig, 2*pi/8)
            assert np.allclose(diag_19_neig, 0.9553166181245093)
        else:
            assert np.any(np.isclose(side_len*curvature_factor, real_dist, atol=0.002))


        # lengths of border areas
        nonzero_borders = my_grid.get_cell_borders().data
        borders = my_grid.get_cell_borders().toarray()

        if i==0:
            # borders are always quaters of the big circle
            border_len = 2*pi / 4
            assert np.allclose(nonzero_borders, border_len)
        elif i==1:
            # should be close to 1/8 of the big circle, but there are some tiny diag areas too
            diag_borders_19 = [borders[19][4], borders[19][5], borders[19][0], borders[19][3]]
            straight_borders_19 = [borders[19][17], borders[19][15], borders[19][14], borders[19][9]]
            assert np.allclose(diag_borders_19,  0.12089407013591315)
            assert np.allclose(straight_borders_19, 0.5712296571729331)
            triangle_borders_4 = [borders[4][20], borders[4][9], borders[4][17]]
            assert np.allclose(triangle_borders_4, 0.75195171)

    # for N in (40, 272):
    #     my_grid = SphereGridFactory.create(N=N, alg_name="fulldiv", dimensions=4, use_saved=USE_SAVED)
    #     approx_areas = my_grid.get_voronoi_areas(approx=True, using_detailed_grid=True)
    #     theo_area = hypersphere_surface / N
    #     print(np.average(approx_areas), theo_area)
    #     print(f"Warning! There is a decent error in volumes of hypersphere volumes, {np.sum(approx_areas)}!={hypersphere_surface}")
    #     #assert np.isclose(np.sum(approx_areas), hypersphere_surface, atol=0.1, rtol=0.1), f"{np.sum(approx_areas)}
    #     # {hypersphere_surface}"
    #     assert np.allclose(approx_areas, theo_area, atol=0.1, rtol=0.1)
    #     print(my_grid.polytope._get_count_of_point_categories(), np.unique(np.round(approx_areas, 5),
    #                      return_counts=True))
    # TODO: check that groups of volumes = groups of points


def test_voronoi_visual_inspection():
    """
    This function tests that for 3D points on the sphere:
    1) the size of voronoi cells, border areas and distances between centers is as expected from visual inspection
    2) the neigbouring relations are as expected from visual inspection
    3) default function for calculating areas gives values that sum up well

    This doesn't test different ways of calculating areas, or compare with polytope distances; see
    test_voronoi_exact_divisions
    """
    sphere = SphereGrid3DFactory.create("ico", 20, use_saved=False)
    # if you wanna visualize it, uncomment
    # from molgri.plotting.spheregrid_plots import SphereGridPlot
    # import matplotlib.pyplot as plt
    # sg = SphereGridPlot(sphere)
    # sg.plot_grid(save=False, labels=True)
    # sg.plot_voronoi(ax=sg.ax, fig=sg.fig, save=False, labels=True, animate_rot=False)
    # plt.show()

    # test_adjacency
    adj_sphere = sphere.get_voronoi_adjacency().toarray()
    visual_neigh_of_0 = [2, 6, 7, 9, 15]
    visual_neigh_of_3 = [4, 7, 8, 10, 11, 18, 19]
    visual_neigh_of_19 = [3, 7, 8, 14]
    assert np.all(visual_neigh_of_0 == np.nonzero(adj_sphere[0])[0])
    assert np.all(visual_neigh_of_3 == np.nonzero(adj_sphere[3])[0])
    assert np.all(visual_neigh_of_19 == np.nonzero(adj_sphere[19])[0])

    # test areas
    # visual_intuition:
    # small areas: 5, 12, 13, 14, 15, 16, 17, 19
    # large areas: 0, 1, 2, 3, 9, 11
    # sum of areas: 4pi
    areas = sphere.get_cell_volumes()
    expected_largest = areas[2]
    expected_smallest = areas[17]
    expected_small = areas[[5, 12, 13, 14, 15, 16, 17, 19]]
    expected_large = areas[[0, 1, 2, 3, 9, 11]]
    assert np.all([x > y for x in expected_large for y in expected_small])
    assert np.isclose(np.sum(areas), 4*pi)
    # the largest one is almost  like there would be only one division
    assert np.all(expected_largest >= areas)
    assert np.isclose(expected_largest, 4*pi/12, atol=4e-2, rtol=1e-3)
    # the smallest one ist't negative or zero or sth weird like that
    assert np.all(expected_smallest <= areas)
    assert expected_smallest > 0.2

    # test border lengths
    # visual intuition:
    # expected_long: [13, 15], [18, 7], [0, 2]
    # expected short: [4, 10], [7, 0], [5, 4], [14, 1]
    # no border: [4, 4], [0, 17], [14, 7], [16, 12]
    borders_sparse = sphere.get_cell_borders()
    # the smallest one in sparse ist't negative or zero or sth weird like that
    assert np.min(borders_sparse.data) > 0.1
    borders = borders_sparse.toarray()
    assert issymmetric(borders)
    # visually, an average border is about 10% of a full circle
    avg_len = borders_sparse.sum()/borders_sparse.count_nonzero()
    assert np.isclose(avg_len, 2*pi/10, atol=0.02, rtol=0.05)
    long_borders = np.array([borders[coo[0]][coo[1]] for coo in [(13, 5), (18, 7), (0, 2)]])
    short_borders = np.array([borders[coo[0]][coo[1]]for coo in [[4, 10], [7, 0], [5, 4], [14, 1]]])
    no_borders = np.array([borders[coo[0]][coo[1]] for coo in [[4, 4], [0, 17], [14, 7], [16, 12]]])
    assert np.allclose(no_borders, 0)
    assert np.all([x > y for x in long_borders for y in short_borders])
    assert np.all(0.1 < short_borders)
    assert np.all(short_borders < 0.5)
    assert np.all(0.7 < long_borders)
    assert np.all(long_borders < 1)

    # test lengths between points
    # visual intuition:
    # expected_long: [1, 5], [7, 0], [0, 2], [4, 5]
    # expected short: [17, 10], [6, 16], [18, 10], [18, 15]
    # no border: [7, 15], [6, 5], [15, 15], [18, 0]
    distances_sparse = sphere.get_center_distances()
    assert np.min(distances_sparse.data) > 0.4
    # the smallest one in sparse ist't negative or zero or sth weird like that
    distances = distances_sparse.toarray()
    assert issymmetric(distances)
    # visually, an average border is about 1/7 of a full circle
    avg_len = distances_sparse.sum()/distances_sparse.count_nonzero()
    assert np.isclose(avg_len, 2*pi/7, atol=0.02, rtol=0.05)
    long_distances = np.array([distances[coo[0]][coo[1]] for coo in [[1, 5], [7, 0], [0, 2], [4, 5]]])
    short_distances = np.array([distances[coo[0]][coo[1]]for coo in [[17, 10], [6, 16], [18, 10], [18, 15]]])
    no_distances = np.array([distances[coo[0]][coo[1]] for coo in [[7, 15], [6, 5], [15, 15], [18, 0]]])
    assert np.allclose(no_distances, 0)
    assert np.all([x > y for x in long_distances for y in short_distances])
    assert np.all(0.5 < short_distances)
    assert np.all(short_distances < 0.65)
    assert np.all(1 < long_distances)
    assert np.all(long_distances < 1.2)

def test_full_and_half_hypersphere():
    for alg in ["cube4D", "randomQ"]:
        for N in [8, 15, 73]:
            hypersphere = SphereGrid4DFactory.create(alg, N, use_saved=False)
            half_grid = hypersphere.get_grid_as_array()
            full_grid = hypersphere.get_full_hypersphere_array()
            assert np.allclose(full_grid[:N], half_grid), f"{alg}, {full_grid[:N]}, {half_grid}"
            assert two_sets_of_quaternions_equal(full_grid[:N], full_grid[N:])
            # no repeating rows in first and second half
            for el in full_grid[:N]:
                assert len(np.nonzero(np.all(np.isclose(el, full_grid[N:]), axis=1))[0])==0


def test_hypersphere_adj():
    # the % of neighbours must be the same whether using all points or half points & opposing neighbours
    for alg in ["cube4D", "randomQ"]:
        for N in [8, 15, 73]:
            hypersphere = SphereGrid4DFactory.create(alg, N, use_saved=False)
            hypersphere.get_voronoi_adjacency()

if __name__ == "__main__":
    # test_saving_rotobj()
    # test_general_grid_properties()
    # test_statistics()
    # test_ordering()
    # test_voronoi_exact_divisions()
    # test_voronoi_visual_inspection()
    test_full_and_half_hypersphere()
