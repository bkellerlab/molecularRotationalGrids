from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.linalg import issymmetric

from molgri.constants import DEFAULT_ALGORITHM_B, DEFAULT_ALGORITHM_O, GRID_ALGORITHMS_3D
from molgri.space.rotobj import SphereGrid3DFactory, SphereGrid4DFactory, SphereGridFactory, SphereGridNDim
from molgri.space.utils import k_is_a_row
from molgri.space.voronoi import RotobjVoronoi, HalfRotobjVoronoi
from tests.test_rotobj import USE_SAVED


SPHERE_SURFACE = 4*pi
HYPERSPHERE_SURFACE = 2* pi ** 2


def example_rotobj(dim: int, sizes=None, half=False, detailed=True) -> Tuple[SphereGridNDim, RotobjVoronoi]:
    assert dim == 3 or dim == 4
    if sizes is None and dim == 3:
        # two exact divisions and one non-exact
        sizes = [12, 42, 85,]
    elif sizes is None and dim == 4:
        sizes = [16, 80, 544]
    for size in sizes:
        if dim == 3:
            my_rotobj = SphereGrid3DFactory().create(alg_name=DEFAULT_ALGORITHM_O, N=size, use_saved=False)
        else:
            my_rotobj = SphereGrid4DFactory().create(alg_name=DEFAULT_ALGORITHM_B, N=size, use_saved=False)

        # create a full Voronoi grid, then optionally get a derived half grid
        my_voronoi = RotobjVoronoi(my_rotobj.get_grid_as_array(only_upper=False), using_detailed_grid=detailed)
        if half:
            my_voronoi = my_voronoi.get_related_half_voronoi()
        yield my_rotobj, my_voronoi


def _assert_volumes_make_sense(volumes, dim, half, approx):
    N = len(volumes)

    if dim == 3:
        volume_sum = SPHERE_SURFACE
    elif dim == 4:
        volume_sum = HYPERSPHERE_SURFACE
    else:
        raise ValueError("dim must be 3 or 4")
    if half:
        volume_sum /= 2
    theo_volume = volume_sum / N

    if N == 85:
        atol = 3
        rtol = 0.3
        # if not perfect division, check out largest/smallest volumes
        expected_small = volumes[[5, 12, 13, 14, 15, 16, 17, 19]]
        expected_large = volumes[[0, 1, 2, 3, 9, 11]]
        print(expected_small, expected_large, np.average(volumes), np.argmin(volumes), np.argmax(volumes))
        assert np.all([x > y for x in expected_large for y in expected_small])
    # assert np.isclose(np.sum(areas), 4 * pi), f"{np.sum(areas)}!={4 * pi}"
    # # the largest one is almost  like there would be only one division
    # assert np.all(expected_largest >= areas)
    # assert np.isclose(expected_largest, 4 * pi / 12, atol=4e-2, rtol=1e-3)
    # # the smallest one ist't negative or zero or sth weird like that
    # assert np.all(expected_smallest <= areas)
    # assert expected_smallest > 0.2
    else:
        # less exact if approx or a small num of points
        if N < 15 or approx:
            atol = 0.01
            rtol = 0.15
        else:
            atol = 0.01,
            rtol = 0.05

    # assert average close to theoretical
    print(volumes, theo_volume)
    assert np.isclose(np.average(volumes), theo_volume, atol=atol, rtol=rtol)

    # sum of approximate areas also close to theoretical
    assert np.isclose(np.sum(volumes), volume_sum, atol=atol, rtol=rtol), f"{np.sum(volumes), volume_sum, approx}"

    # all values are reasonably close to each other
    if N < 10 or N==85:
        assert np.std(volumes) < 0.13
    else:
        assert np.std(volumes) < 0.08

    # no super weird values
    assert np.all(volumes < 10*theo_volume)
    assert np.all(volumes > 0.09*theo_volume)


def test_reduced_coordinates():
    """
    This function tests that reduced coordinates still map to exactly the same vertices (only removing non-unique ones)
    """
    for half in [False, True]:
        for dim in [3, 4]:
            my_example = example_rotobj(dim=dim, half=half)
            for el in my_example:
                my_rotobj, my_voronoi = el
                all_vertices = my_voronoi.get_all_voronoi_vertices(reduced=False)
                reduced_vertices = my_voronoi.get_all_voronoi_vertices(reduced=True)
                all_regions = my_voronoi.get_all_voronoi_regions(reduced=False)
                reduced_regions = my_voronoi.get_all_voronoi_regions(reduced=True)
                assert len(reduced_regions) == len(all_regions), "No point should get lost when redundant vertices are " \
                                                                 "removed"
                for i, region in enumerate(all_regions):
                    i_before_reduction = region
                    i_after_reduction = reduced_regions[i]
                    vertices_before_reduction = all_vertices[i_before_reduction]
                    unique_before = np.unique(vertices_before_reduction, axis=0)
                    vertices_after_reduction = reduced_vertices[i_after_reduction]
                    # test: all rows in vertices_after_reduction must be present in vertices_before_reduction
                    assert len(i_before_reduction) == len(i_after_reduction)
                    for row in vertices_after_reduction:
                        assert k_is_a_row(unique_before, row), f"{unique_before, row}"
                    # test: all rows in vertices_after_reduction must be present in vertices_before_reduction
                    # but some may repeat, so we need to do unique in both cases
                    unique_before = np.unique(vertices_before_reduction, axis=0)
                    unique_after = np.unique(vertices_after_reduction, axis=0)
                    assert np.allclose(unique_before, unique_after)


def _assert_neig_of_ico_12(obtained_adj_mat: NDArray):
    expected_neig_zero = {10, 7, 9, 2, 6}
    expected_neig_six = {0, 2, 5, 4, 10}
    expected_neig_eight = {1, 11, 3, 7, 9}
    assert set(np.nonzero(obtained_adj_mat[0])[0]) == expected_neig_zero
    assert set(np.nonzero(obtained_adj_mat[6])[0]) == expected_neig_six
    assert set(np.nonzero(obtained_adj_mat[8])[0]) == expected_neig_eight


def _assert_neig_of_ico_42(obtained_adj_mat: NDArray):
    expected_neig_zero = {26, 20, 30, 15, 27}
    expected_neig_forty = {16, 4, 41, 38, 10, 17}
    expected_neig_fourteen = {24, 37, 8, 9, 19, 22}
    assert set(np.nonzero(obtained_adj_mat[0])[0]) == expected_neig_zero
    assert set(np.nonzero(obtained_adj_mat[40])[0]) == expected_neig_forty
    assert set(np.nonzero(obtained_adj_mat[14])[0]) == expected_neig_fourteen


def _assert_neig_of_ico_85(obtained_adj_mat: NDArray):
    expected_neig_4 = {16, 76, 40, 50, 41, 69, 33, 31}
    expected_neig_68 = {24, 1, 37, 75}  # questionable: 14
    expected_neig_26 = {7, 53, 52, 0, 54, 73}  #  questionable: 15, 22
    assert set(np.nonzero(obtained_adj_mat[4])[0]) == expected_neig_4
    assert set(np.nonzero(obtained_adj_mat[68])[0]) == expected_neig_68
    assert set(np.nonzero(obtained_adj_mat[26])[0]) == expected_neig_26


def test_rotobj():
    """
    This function tests that the half/full sphere Rotobj in 3D:
    1) give the right cell volumes, whether approximations or not, detailed or not
    2) give the right adjacency matrix
    3) give the right border lengths and distances between centers
    """
    for half in [False, True]:
        for detailed in [True, False]:
            my_example = example_rotobj(dim=3, half=half, detailed=detailed)
            for el in my_example:
                my_rotobj, my_voronoi = el
                print(detailed, half, my_rotobj.N)
                # plotting - don't wait for all to finish, very long
                # from molgri.plotting.voronoi_plots import VoronoiPlot
                # import matplotlib.pyplot as plt
                # vp = VoronoiPlot(my_voronoi)
                # vp.plot_centers(save=False)
                # vp.plot_vertices(ax=vp.ax, fig=vp.fig, save=False, alpha=0.5, labels=False)
                # vp.plot_borders(ax=vp.ax, fig=vp.fig, save=False, animate_rot=False, reduced=True)
                # vp.plot_vertices_of_i(75, ax=vp.ax, fig=vp.fig, save=False)
                # vp.plot_vertices_of_i(36, ax=vp.ax, fig=vp.fig, save=False)
                # plt.show()
                # testing volumes
                for approx in [False, True]:
                    all_volumes = my_voronoi.get_voronoi_volumes(approx=approx)
                    # tests on volumes
                    _assert_volumes_make_sense(volumes=all_volumes, dim=3, half=half, approx=approx)
                # testing adjacency matrix
                adj_matrix = my_voronoi.get_voronoi_adjacency().toarray()
                if my_rotobj.N == 12 and not half:
                    _assert_neig_of_ico_12(adj_matrix)
                elif my_rotobj.N == 42 and not half:
                    _assert_neig_of_ico_42(adj_matrix)
                elif my_rotobj.N == 85 and not half:
                    _assert_neig_of_ico_85(adj_matrix)


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


def test_3D_voronoi_visual_inspection():
    """
    This function tests that for 3D points on the sphere:
    1) the size of voronoi cells, border areas and distances between centers is as expected from visual inspection
    2) the neighbouring relations are as expected from visual inspection
    3) default function for calculating areas gives values that sum up well

    This doesn't test different ways of calculating areas, or compare with polytope distances; see
    test_voronoi_exact_divisions
    """
    sphere = SphereGrid3DFactory.create("ico", 20, use_saved=False)
    # if you wanna visualize it, uncomment
    # from molgri.plotting.spheregrid_plots import SphereGridPlot
    # import matplotlib.pyplot as plt
    # sg = SphereGridPlot(sphere)
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
    assert np.isclose(np.sum(areas), 4*pi), f"{np.sum(areas)}!={4*pi}"
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


def test_4Dvoronoi():
    """
    This function tests that for 4D points on the sphere:
    1) the size of voronoi cells, border areas and distances between centers are reasonable and don't have huge
    deviations
    2) the neighbouring relations are as expected from visual inspection (for cube4D)
    3) default function for calculating areas gives values that sum up well for full and half hyperspheres
    """
    # 8, 40, 250
    for N in [8, ]:  #40, 250
        hypersphere = SphereGrid4DFactory.create(DEFAULT_ALGORITHM_B, N, use_saved=False)
        # uncomment for plotting
        from molgri.plotting.spheregrid_plots import EightCellsPlot, SphereGridPlot
        sp = SphereGridPlot(hypersphere)
        # sp.plot_voronoi(points=True, vertices=True, borders=False, animate_rot=True)
        # sp.plot_adjacency_array()
        # sp.plot_center_distances_array(only_upper=False)
        # sp.plot_cdist_array(only_upper=False)
        #sp.plot_center_distances_array()
        ep = EightCellsPlot(hypersphere.polytope, only_half_of_cube=False)
        ep.make_all_8cell_neighbours(node_index=15, animate_rot=False)
        adj_array = hypersphere.get_voronoi_adjacency(only_upper=False, include_opposing_neighbours=False).toarray()
        print(adj_array.astype(int))
        avg_neigh1 = np.mean(np.nansum(adj_array, axis=0))
        adj_array3 = hypersphere.get_voronoi_adjacency(only_upper=True, include_opposing_neighbours=True).toarray()
        avg_neigh3 = np.mean(np.nansum(adj_array3, axis=0))
        assert np.isclose(avg_neigh1, avg_neigh3, atol=0.2, rtol=0.01)
        # volumes
        all_volumes = hypersphere.get_cell_volumes(approx=True, only_upper=False)
        half_volumes = hypersphere.get_cell_volumes(approx=True, only_upper=True)
        # first half of all_volumes are exactly the half_volumes
        assert 2 * len(half_volumes) == len(all_volumes)
        assert np.allclose(all_volumes[:len(half_volumes)], half_volumes)
        # second half of all_volumes are also at least approx the same
        assert np.allclose(all_volumes[len(half_volumes):], half_volumes, atol=0.3, rtol=0.05)
        # sum is approx half of the hypersphere (with enough points):
        if N > 10:
            assert np.isclose(np.sum(half_volumes), pi ** 2, atol=0.6, rtol=0.1), f"{np.sum(half_volumes)}!={pi ** 2}"


def test_hypersphere_adj():
    # the % of neighbours must be the same whether using all points or half points & opposing neighbours
    my_example = example_rotobj(dim=3, half=False, sizes=(13,))
    for el in my_example:
        my_rotobj, my_voronoi = el
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        #print(my_voronoi.get_all_voronoi_centers())
        ax.scatter(*my_voronoi.get_all_voronoi_centers().T, color="black")
        # average number of neighbours per row
        print(np.average(np.sum(my_voronoi.get_voronoi_adjacency().toarray(), axis=0)))
        #print(np.average(my_voronoi.get_voronoi_adjacency().toarray()))
    my_example = example_rotobj(dim=3, half=True, sizes=(13,))
    for el in my_example:
        my_rotobj, my_voronoi = el
        ax.scatter(*my_voronoi.get_all_voronoi_centers().T, color="red", marker="x")
        plt.show()
        # average number of neighbours per row
        print(np.average(np.sum(my_voronoi.get_voronoi_adjacency().toarray(), axis=0)))


def test_full_and_half():
    """
    Tests that for 4D voronoi grids, the half grids have
    0) N centers and N volumes, NxN adjacency grids
    1) exactly the same vertices and points, just not all of them
    2) the same average number of neighbours
    3) the same volumes (again just not all of them)
    4) TODO: and same lengths, borders
    """
    dim = 4 # will only be using this in 4D
    for size in [29, 55]:
        my_example_full = example_rotobj(dim=dim, half=False, sizes=[size,])
        my_example_half = example_rotobj(dim=dim, half=True, sizes=[size, ])
        for el_full, el_half in zip(my_example_full, my_example_half):
            my_rotobj_full, my_voronoi_full = el_full
            my_rotobj_half, my_voronoi_half = el_half
            # all half points in full points (but not vice versa)
            half_points = my_voronoi_half.get_all_voronoi_centers()
            all_points = my_voronoi_full.get_all_voronoi_centers()
            # IN 4D, full vronoi grid has too many (2N) points, the ones that really interest us are the first N points
            assert len(half_points) == size
            assert len(all_points) == 2*size
            assert np.allclose(half_points, all_points[:size])
            for hp in half_points:
                assert k_is_a_row(all_points, hp)

            # all half vertices in full vertices (but not vice versa)
            half_points = my_voronoi_half.get_all_voronoi_vertices()
            all_points = my_voronoi_full.get_all_voronoi_vertices()
            assert len(half_points) < len(all_points)
            for hp in half_points:
                assert k_is_a_row(all_points, hp)

            # test adjacency, lengths, borders
            for my_property in ["adjacency", "center_distances", "border_len"]:
                adj_full = my_voronoi_full._calculate_N_N_array(property=my_property).toarray()
                adj_half = my_voronoi_half._calculate_N_N_array(property=my_property).toarray()
                assert adj_full.shape == (2*size, 2*size)
                assert adj_half.shape == (size, size)
                # num of neighbours on average same in full sphere and half sphere with opposite neighbours
                print(size, my_property, np.average(np.sum(adj_full, axis=0)), np.average(np.sum(adj_half, axis=0)))

                print()

                assert np.isclose(np.average(np.sum(adj_full, axis=0)), np.average(np.sum(adj_half, axis=0)), atol=0.2,
                                  rtol=0.01)
                # if you select only_upper=False, include_opposing_neighbours=False, it's the same as full grid
                option2 = my_voronoi_half._calculate_N_N_array(property=my_property,
                                                               only_upper=False,
                                                               include_opposing_neighbours=False).toarray()
                assert np.allclose(option2, adj_full)
                # if you select only_upper=True, include_opposing_neighbours=False, you should get the upper left
                # corner of full
                # adjacency array
                option3 = my_voronoi_half._calculate_N_N_array(property=my_property, only_upper=True,
                                                                include_opposing_neighbours=False).toarray()
                assert np.allclose(option3, adj_full[:size, :size])

            # volumes
            full_volumes = my_voronoi_full.get_voronoi_volumes()
            half_volumes = my_voronoi_half.get_voronoi_volumes()
            assert len(full_volumes) == 2*size
            assert len(half_volumes) == size
            assert np.allclose(half_volumes, full_volumes[:size])


if __name__ == "__main__":
    for dim in [4, 3]:
        my_example = example_rotobj(dim=dim, half=False, sizes=[13,])
        for el in my_example:
            my_rotobj, my_voronoi = el
            my_voronoi.get_cell_borders()

    # test_reduced_coordinates()
    test_full_and_half()

    # test_rotobj()
    # test_voronoi_exact_divisions()
    # test_3D_voronoi_visual_inspection()
    # test_4Dvoronoi()
    # test_hypersphere_adj()
