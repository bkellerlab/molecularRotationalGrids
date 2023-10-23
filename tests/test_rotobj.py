import pandas as pd
from molgri.assertions import all_row_norms_equal_k
from tqdm import tqdm
from scipy.constants import pi

from molgri.space.rotobj import SphereGridFactory
from molgri.constants import GRID_ALGORITHMS_3D, GRID_ALGORITHMS_4D, DEFAULT_ALPHAS_3D

import numpy as np

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


def test_voronoi_areas():
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
            my_areas = my_grid.get_voronoi_areas()
            # Voronoi areas can be calculated for all algorithms, they have the right length, are + and add up to 4pi
            assert len(my_areas) == 45
            assert np.all(my_areas > 0)
            assert np.isclose(np.sum(my_areas), sphere_surface)
        except ValueError:
            print(f"Duplicate generator issue for {name}.")

    for name in ["ico", "cube3D"]:
        for N in (62, 114):
            my_grid = SphereGridFactory.create(N=N, alg_name=name, dimensions=3, use_saved=USE_SAVED)
            exact_areas = my_grid.get_voronoi_areas(approx=False)
            approx_areas = my_grid.get_voronoi_areas(approx=True, using_detailed_grid=True)
            # for cube3D and ico, approximate areas are close to exact areas
            assert np.allclose(exact_areas, approx_areas, atol=1e-2, rtol=0.1)
            # each surface individually should be somewhat close to theoretical
            theo_area = sphere_surface / N
            assert np.isclose(np.average(approx_areas), theo_area, atol=0.01, rtol=0.1)
            # sum of approximate areas also close to theoretical
            assert np.isclose(np.sum(approx_areas), sphere_surface, atol=0.1, rtol=0.1)
            # even if not using detailed grid, the values should be somewhat close
            not_detailed = my_grid.get_voronoi_areas(approx=True, using_detailed_grid=False)
            assert np.allclose(not_detailed, exact_areas, atol=0.1, rtol=0.1)

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

if __name__ == "__main__":
    test_saving_rotobj()
    test_general_grid_properties()
    test_statistics()
    test_ordering()
    test_voronoi_areas()
