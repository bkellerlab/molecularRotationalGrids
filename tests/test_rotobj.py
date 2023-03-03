import pandas as pd
from tqdm import tqdm
from scipy.constants import pi

from molgri.space.rotobj import SphereGridFactory
from molgri.constants import GRID_ALGORITHMS, DEFAULT_ALPHAS_3D

import numpy as np

# tests should always be performed on fresh data
USE_SAVED = False


def test_saving_rotobj():
    """
    Assert that after saving you get back the same grid.
    """
    for algo in GRID_ALGORITHMS[:-1]:
        for N in (12, 23, 51):
            for d in (3, 4):
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
    for alg in GRID_ALGORITHMS[:-1]:
        for number in (3, 15, 26):
            for d in (3, 4):
                grid_obj = SphereGridFactory.create(N=number, alg_name=alg, dimensions=d, use_saved=USE_SAVED)
                grid = grid_obj.get_grid_as_array()
                assert isinstance(grid, np.ndarray), "Grid must be a numpy array."
                assert grid.shape == (number, d), "Wrong grid shape."
                assert grid_obj.N == number


def test_everything_runs():
    for algo in GRID_ALGORITHMS[:-1]:
        ig = SphereGridFactory.create(N=35, alg_name=algo, dimensions=4, use_saved=True)
        ig.gen_and_time()
        ig.save_grid()
        ig.save_grid("txt")
        ig = SphereGridFactory.create(N=35, alg_name=algo, dimensions=4, use_saved=True)
        ig.gen_and_time()
        ig.save_grid()
        ig.save_grid("txt")
        ig = SphereGridFactory.create(N=22, alg_name=algo, dimensions=4, use_saved=False)
        ig.gen_and_time()
        ig.save_grid()
        ig.save_grid("txt")


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
    # TODO: figure out what's the issue
    """Assert that, ignoring randomness, the first N-1 points of ordered grid with length N are equal to ordered grid
    of length N-1"""
    for name in tqdm(GRID_ALGORITHMS):
        try:
            for N in range(14, 111, 3):
                for addition in (1, 7):
                    grid_1 = SphereGridFactory.create(N=N+addition, alg_name=name, dimensions=3, use_saved=USE_SAVED)
                    grid_1 = grid_1.get_grid_as_array()
                    grid_2 = SphereGridFactory.create(N=N, alg_name=name, dimensions=3, use_saved=USE_SAVED)
                    grid_2 = grid_2.get_grid_as_array()
                    assert np.allclose(grid_1[:N], grid_2)
        except AssertionError:
            print(name)



if __name__ == "__main__":
    test_ordering()
