import pandas as pd
import numpy as np
from scipy.constants import pi

from molgri.space.rotobj import SphereGrid4DFactory, SphereGridFactory
from molgri.constants import GRID_ALGORITHMS_3D, GRID_ALGORITHMS_4D, DEFAULT_ALPHAS_3D
from molgri.space.utils import all_row_norms_equal_k, two_sets_of_quaternions_equal

USE_SAVED = False

# don't test fulldiv
GRID_ALGORITHMS_4D = [x for x in GRID_ALGORITHMS_4D if x != "fulldiv"]


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


def test_full_and_half_hypersphere():
    """
    This function tests:
    1) for 4D grids, get_grid returns a all q on upper and then symmetric -q on bottom side of the hypersphere
    """
    for alg in ["cube4D", "randomQ"]:
        for N in [8, 15, 73]:
            hypersphere = SphereGrid4DFactory.create(alg, N, use_saved=False)
            half_grid = hypersphere.get_grid_as_array(only_upper=True)
            full_grid = hypersphere.get_grid_as_array(only_upper=False)
            assert np.allclose(full_grid[:N], half_grid), f"{alg}, {full_grid[:N]}, {half_grid}"
            assert np.allclose(full_grid[N:], -half_grid), f"{alg}, {full_grid[:N]}, {half_grid}"
            assert two_sets_of_quaternions_equal(full_grid[:N], full_grid[N:])
            # no repeating rows in first and second half
            for el in full_grid[:N]:
                assert len(np.nonzero(np.all(np.isclose(el, full_grid[N:]), axis=1))[0]) == 0


if __name__ == "__main__":
    test_saving_rotobj()
    test_general_grid_properties()
    test_statistics()
    test_ordering()
    test_full_and_half_hypersphere()
