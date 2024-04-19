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
        sizes = [16, 55, 80]
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


def test_rotobj_voronoi_3D_exact():
    """
    This function tests that the full sphere Rotobj in 3D with exact division of sides:
    1) give the right cell volumes, whether approximations or not, detailed or not
    2) give the right adjacency matrix
    3) TODO: give the right border lengths and distances between centers
    """
    for N in [12, 42, 162]: # 162
        my_grid = SphereGridFactory.create(N=N, alg_name=DEFAULT_ALGORITHM_O, dimensions=3, use_saved=False)
        my_voronoi = RotobjVoronoi(my_grid.get_grid_as_array(only_upper=False), using_detailed_grid=True)
        # plotting
        # from molgri.plotting.voronoi_plots import VoronoiPlot
        # import matplotlib.pyplot as plt
        # vp = VoronoiPlot(my_voronoi)
        # vp.plot_centers(save=False)
        # vp.plot_vertices(ax=vp.ax, fig=vp.fig, save=False, alpha=0.5, labels=False)
        # vp.plot_borders(ax=vp.ax, fig=vp.fig, save=False, animate_rot=False, reduced=True)
        # plt.show()
        # testing volumes
        for approx in [False, True]:
            if approx:
                atol = 0.03
                rtol = 0.05
            else:
                atol = 1e-8
                rtol = 1e-5

            all_volumes = my_voronoi.get_voronoi_volumes(approx=approx)
            # average volume
            assert np.isclose(SPHERE_SURFACE / N, np.average(all_volumes), atol=atol, rtol=rtol)
            # sum of volumes
            assert np.isclose(SPHERE_SURFACE, np.sum(all_volumes), atol=atol, rtol=rtol)
            # very little deviation
            assert np.std(all_volumes) < 0.02
            # no weird values
            assert np.all(all_volumes < 1.3 * SPHERE_SURFACE / N)
            assert np.all(all_volumes > 0.7 * SPHERE_SURFACE / N)

        # testing adjacency matrix
        adj_matrix = my_voronoi.get_voronoi_adjacency().toarray()
        if my_grid.N == 12:
            expected_neig_zero = {10, 7, 9, 2, 6}
            expected_neig_six = {0, 2, 5, 4, 10}
            expected_neig_eight = {1, 11, 3, 7, 9}
            assert set(np.nonzero(adj_matrix[0])[0]) == expected_neig_zero
            assert set(np.nonzero(adj_matrix[6])[0]) == expected_neig_six
            assert set(np.nonzero(adj_matrix[8])[0]) == expected_neig_eight
        elif my_grid.N == 42:
            expected_neig_zero = {26, 20, 30, 15, 27}
            expected_neig_forty = {16, 4, 41, 38, 10, 17}
            expected_neig_fourteen = {24, 37, 8, 9, 19, 22}
            assert set(np.nonzero(adj_matrix[0])[0]) == expected_neig_zero
            assert set(np.nonzero(adj_matrix[40])[0]) == expected_neig_forty
            assert set(np.nonzero(adj_matrix[14])[0]) == expected_neig_fourteen
        # testing distances
        cd = my_voronoi.get_center_distances().toarray()
        unique_cd, cd_counts = np.unique(cd, return_counts=True)
        if my_grid.N == 12:
            # 12 points, 5 neighbours each = 60 distances
            assert np.sum(cd_counts[1:]) == 60
            # all exactly same length
            assert len(cd_counts) == 2
            # length approx 1/6 of full circle
            assert np.allclose(cd, 2 * pi / 6, atol=0.7, rtol=0.35)
        elif my_grid.N == 42:
            assert np.sum(cd_counts[1:]) == 240
            # length approx 1/12 of full circle
            assert np.allclose(cd[cd>0], 2 * pi / 12, atol=0.1, rtol=0.05)
        # testing border areas
        bor = my_voronoi.get_cell_borders().toarray()
        unique_bor, bor_counts = np.unique(bor, return_counts=True)
        if my_grid.N == 12:
            # 12 points, 5 neighbours each = 60 borders
            assert np.sum(bor_counts[1:]) == 60
            assert np.allclose(np.average(unique_bor[1:]), 0.7297276562269662)
        elif my_grid.N == 42:
            assert np.sum(bor_counts[1:]) == 240
            assert np.allclose(np.average(unique_bor[1:]), 0.3533246808973012)



def test_rotobj_voronoi_3D_non_exact():
    """
    This function tests that for 3D points on the sphere:
    1) the size of voronoi cells, border areas and distances between centers is as expected from visual inspection
    2) the neighbouring relations are as expected from visual inspection
    3) default function for calculating areas gives values that sum up well

    This doesn't test different ways of calculating areas, or compare with polytope distances; see
    test_voronoi_exact_divisions
    """
    sphere = SphereGrid3DFactory.create("ico", 20, use_saved=False)
    my_voronoi = RotobjVoronoi(sphere.get_grid_as_array(only_upper=False), using_detailed_grid=True)
    # if you wanna visualize it, uncomment
    # from molgri.plotting.voronoi_plots import VoronoiPlot
    # import matplotlib.pyplot as plt
    # vp = VoronoiPlot(my_voronoi)
    # vp.plot_centers(save=False)
    # vp.plot_vertices(ax=vp.ax, fig=vp.fig, save=False, alpha=0.5, labels=False)
    # vp.plot_borders(ax=vp.ax, fig=vp.fig, save=False, animate_rot=False, reduced=True)
    # plt.show()

    # test_adjacency
    adj_sphere = my_voronoi.get_voronoi_adjacency().toarray()
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
    areas = my_voronoi.get_voronoi_volumes()
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
    # expected_long: [13, 5], [18, 7], [0, 2]
    # expected short: [4, 10], [7, 0], [5, 4], [14, 1]
    # no border: [4, 4], [0, 17], [14, 7], [16, 12]
    borders_sparse = my_voronoi.get_cell_borders()
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
    distances_sparse = my_voronoi.get_center_distances()
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


def test_rotobj_voronoi_4D():
    """
    This function tests that for 4D points on the (half)hypersphere:
    1) the size of voronoi cells, TODO: border areas and distances between centers are reasonable and don't have huge
    deviations
    2) the neighbouring relations are as expected from visual inspection (for cube4D)
    3) default function for calculating areas gives values that sum up well for full and half hyperspheres
    """
    # 8, 40, 250
    for N in [8, 40, 272]:
        hypersphere = SphereGrid4DFactory.create(DEFAULT_ALGORITHM_B, N, use_saved=False)
        # uncomment for plotting
        from molgri.plotting.spheregrid_plots import EightCellsPlot, SphereGridPlot
        my_voronoi = RotobjVoronoi(hypersphere.get_grid_as_array(only_upper=False), using_detailed_grid=True)
        half_voronoi = my_voronoi.get_related_half_voronoi()

        # plotting voronoi
        # from molgri.plotting.voronoi_plots import VoronoiPlot
        # import matplotlib.pyplot as plt
        # vp = VoronoiPlot(my_voronoi)
        # vp.plot_centers(save=False)
        # vp.plot_vertices(ax=vp.ax, fig=vp.fig, save=False, alpha=0.5, labels=False)
        # vp.plot_borders(ax=vp.ax, fig=vp.fig, save=False, animate_rot=False, reduced=True)
        # plt.show()

        # plotting eight cells
        # ep = EightCellsPlot(hypersphere.polytope, only_half_of_cube=False)
        # ep.make_all_8cell_neighbours(node_index=15, animate_rot=False)

        # volumes
        all_volumes = my_voronoi.get_voronoi_volumes()
        half_volumes = half_voronoi.get_voronoi_volumes()
        # sum makes sense
        print(np.sum(all_volumes), HYPERSPHERE_SURFACE)
        assert np.isclose(np.sum(all_volumes), HYPERSPHERE_SURFACE, atol=1, rtol=0.025)
        assert np.isclose(np.sum(half_volumes), HYPERSPHERE_SURFACE/2, atol=0.3, rtol=0.025)
        # averages make sense
        assert np.isclose(np.average(all_volumes), HYPERSPHERE_SURFACE/(2*N), atol=0.3, rtol=0.025)
        assert np.isclose(np.average(half_volumes), HYPERSPHERE_SURFACE/(2*N), atol=0.3, rtol=0.025)


        # adjacency
        full_adjacency = my_voronoi.get_voronoi_adjacency().toarray()
        half_adjacency = half_voronoi.get_voronoi_adjacency().toarray()
        #print(np.unique(full_adjacency, return_counts=True), np.unique(half_adjacency, return_counts=True))
        # average num of neighbours constant
        #print(np.average(np.sum(full_adjacency, axis=0)), np.average(np.sum(half_adjacency, axis=0)))
        # visual inspection of neighbours


        # borders
        full_borders = my_voronoi.get_cell_borders().toarray()
        half_borders = half_voronoi.get_cell_borders().toarray()
        #print(np.unique(full_borders, return_counts=True), np.unique(half_borders, return_counts=True))
        # in 2d the sum of borders basically doubles
        # TODO: check convergence if adding more additional points
        print("4d", N, np.sum(full_borders), np.sum(full_borders)/N, 4*pi, np.sum(full_borders)/4/pi/N)


def test_full_and_half():
    """
    Tests that for 4D voronoi grids, the half grids have
    0) N centers and N volumes, NxN adjacency grids
    1) exactly the same vertices and points, just not all of them
    2) the same average number of neighbours
    3) the same volumes (again just not all of them)
    4) and same lengths, borders

    It DOESN'T check if any way if these values are sensible, only how they change between full grid and half grid
    """
    dim = 4  # will only be using this in 4D
    for size in [29, 55]:
        my_example_full = example_rotobj(dim=dim, half=False, sizes=[size,])
        for el_full in my_example_full:
            my_rotobj_full, my_voronoi_full = el_full
            my_voronoi_half = my_voronoi_full.get_related_half_voronoi()
            # all half points in full points (but not vice versa)
            half_points = my_voronoi_half.get_all_voronoi_centers()
            all_points = my_voronoi_full.get_all_voronoi_centers()
            # IN 4D, full voronoi grid has too many (2N) points, the ones that really interest us are the first N points
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
                adj_full = my_voronoi_full._calculate_N_N_array(sel_property=my_property).toarray()
                adj_half = my_voronoi_half._calculate_N_N_array(sel_property=my_property).toarray()
                assert adj_full.shape == (2*size, 2*size)
                assert adj_half.shape == (size, size)
                # num of neighbours on average same in full sphere and half sphere with opposite neighbours
                assert np.isclose(np.average(np.sum(adj_full, axis=0)), np.average(np.sum(adj_half, axis=0)), atol=0.2,
                                  rtol=0.01)
                # if you select only_upper=False, include_opposing_neighbours=False, it's the same as full grid
                option2 = my_voronoi_half._calculate_N_N_array(sel_property=my_property,
                                                               only_upper=False,
                                                               include_opposing_neighbours=False).toarray()
                assert np.allclose(option2, adj_full)
                # if you select only_upper=True, include_opposing_neighbours=False, you should get the upper left
                # corner of full
                # adjacency array
                option3 = my_voronoi_half._calculate_N_N_array(sel_property=my_property, only_upper=True,
                                                                include_opposing_neighbours=False).toarray()
                assert np.allclose(option3, adj_full[:size, :size])

            # volumes
            full_volumes = my_voronoi_full.get_voronoi_volumes()
            half_volumes = my_voronoi_half.get_voronoi_volumes()
            assert len(full_volumes) == 2*size
            assert len(half_volumes) == size
            assert np.allclose(half_volumes, full_volumes[:size])


if __name__ == "__main__":
    #test_reduced_coordinates()    # done
    #test_full_and_half()   # done
    #test_rotobj_voronoi_3D_non_exact()  # done
    #test_rotobj_voronoi_3D_exact()  # done
    #test_rotobj_voronoi_4D()

    N = 200
    hypersphere = SphereGrid4DFactory.create(DEFAULT_ALGORITHM_B, N, use_saved=False)
    # uncomment for plotting
    from molgri.plotting.spheregrid_plots import EightCellsPlot, SphereGridPlot
    my_voronoi = RotobjVoronoi(hypersphere.get_grid_as_array(only_upper=False), using_detailed_grid=True)
    half_voronoi = my_voronoi.get_related_half_voronoi()

    # plotting voronoi
    # from molgri.plotting.voronoi_plots import VoronoiPlot
    # import matplotlib.pyplot as plt
    # vp = VoronoiPlot(my_voronoi)
    # vp.plot_centers(save=False)
    # vp.plot_vertices(ax=vp.ax, fig=vp.fig, save=False, alpha=0.5, labels=False)
    # vp.plot_borders(ax=vp.ax, fig=vp.fig, save=False, animate_rot=False, reduced=True)
    # plt.show()

    # plotting eight cells
    # ep = EightCellsPlot(hypersphere.polytope, only_half_of_cube=False)
    # ep.make_all_8cell_neighbours(node_index=15, animate_rot=False)

    # volumes
    all_volumes = my_voronoi.get_voronoi_volumes()
    half_volumes = half_voronoi.get_voronoi_volumes()
    # sum makes sense
    print(np.sum(all_volumes), "ideal: ", HYPERSPHERE_SURFACE)
