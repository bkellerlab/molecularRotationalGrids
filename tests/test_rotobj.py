import pandas as pd
import pytest
from scipy.constants import pi
from scipy.spatial import distance_matrix

from molgri.assertions import form_cube
from molgri.space.fullgrid import FullGrid
from molgri.space.rotations import grid2rotation, rotation2grid
from molgri.space.rotobj import build_rotations, build_grid, build_grid_from_name, order_elements
from molgri.constants import GRID_ALGORITHMS

import numpy as np

from molgri.space.utils import normalise_vectors


# tests should always be performed on fresh data
USE_SAVED = False


def test_rotobj2grid2rotobj():
    for algo in GRID_ALGORITHMS[:-1]:
        for N in (12, 23, 51):
            rotobj_start = build_rotations(N, algo, use_saved=False)
            matrices_start = rotobj_start.rotations.as_matrix()
            rotobj_new = build_rotations(N, algo, use_saved=True)
            matrices_new = rotobj_new.rotations.as_matrix()
            assert np.allclose(matrices_start, matrices_new)


def test_grid2rotobj2grid():
    for algo in GRID_ALGORITHMS[:-1]:
        for N in (12, 23, 87, 217):
            rotations_start = build_rotations(N, algo, use_saved=USE_SAVED)
            grid_start_x = rotations_start.grid_x.get_grid()
            grid_start_y = rotations_start.grid_y.get_grid()
            grid_start_z = rotations_start.grid_z.get_grid()
            rotations1 = grid2rotation(grid_start_x, grid_start_y, grid_start_z)
            rotations2 = grid2rotation(grid_start_x, grid_start_y, grid_start_z)
            grid_end1, grid_end2, grid_end3 = rotation2grid(rotations2)
            rotations3 = grid2rotation(grid_end1, grid_end2, grid_end3)
            grid_final1, grid_final2, grid_final3 = rotation2grid(rotations3)
            # rotations remain the same the entire time
            assert np.allclose(rotations_start.rotations.as_matrix(), rotations1.as_matrix())
            assert np.allclose(rotations1.as_matrix(), rotations2.as_matrix())
            assert np.allclose(rotations1.as_matrix(), rotations3.as_matrix())
            # after rotation is once created, everything is deterministic
            assert np.allclose(grid_end1, grid_final1)
            assert np.allclose(grid_end2, grid_final2)
            assert np.allclose(grid_end3, grid_final3)
            # but before ...
            for row1, row2 in zip(grid_end2, grid_start_y):
                try:
                    assert np.allclose(row1, row2)
                except AssertionError:
                    print("y different", row1, row2)
            assert np.allclose(grid_end1, grid_start_x)
            assert np.allclose(grid_end2, grid_start_y)
            assert np.allclose(grid_end3, grid_start_z)


def test_general_grid_properties():
    for alg in GRID_ALGORITHMS[:-1]:
        for number in (3, 15, 26):
            grid_obj = build_grid(number, alg, use_saved=USE_SAVED)
            grid = grid_obj.get_grid()
            assert isinstance(grid, np.ndarray), "Grid must be a numpy array."
            assert grid.shape == (number, 3), "Wrong grid shape."
            assert grid_obj.N == number


def test_cube_3d_grid():
    cube_3d = build_grid(8, "cube3D", use_saved=USE_SAVED)
    grid = cube_3d.get_grid()
    assert form_cube(grid, test_angles=True)


def test_zero_grid():
    ico = build_grid_from_name("ico_0", use_saved=USE_SAVED)
    ico.get_grid()


def test_errors_and_assertions():
    with pytest.raises(ValueError):
        build_grid(15, "icosahedron", use_saved=USE_SAVED)
    with pytest.raises(ValueError):
        build_grid(15, "grid", use_saved=USE_SAVED)
    with pytest.raises(AssertionError):
        build_grid(-15, "ico", use_saved=USE_SAVED)
    with pytest.raises(AssertionError):
        # noinspection PyTypeChecker
        build_grid(15.3, "ico", use_saved=USE_SAVED)
    grid = build_grid(20, "ico", use_saved=USE_SAVED).get_grid()
    with pytest.raises(ValueError):
        order_elements(grid, 25)


def test_everything_runs():
    for algo in GRID_ALGORITHMS[:-1]:
        ig = build_rotations(35, algo, use_saved=True)
        ig.gen_and_time()
        ig.save_all()
        ig.get_grid_z_as_grid().save_grid_txt()
        ig = build_rotations(35, algo, use_saved=True)
        ig.gen_and_time()
        ig.save_all()
        ig.get_grid_z_as_grid().save_grid_txt()
        ig = build_rotations(22, algo, use_saved=False)
        ig.gen_and_time()
        ig.save_all()
        ig.get_grid_z_as_grid().save_grid_txt()


def test_statistics():
    num_points = 35
    num_random = 50
    # grid statistics
    icog = build_grid(num_points, "ico", use_saved=False)
    default_alphas = [pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6]
    icog.save_statistics(num_random=num_random, alphas=default_alphas)
    statistics_csv = pd.read_csv(icog.statistics_path, index_col=0, header=0, dtype=float)
    assert len(statistics_csv) == len(default_alphas)*num_random
    expected_coverages = [0.0669872981077806, 0.2499999999999999,
                          0.4999999999999999, 0.7499999999999999, 0.9330127018922194]
    ideal_coverage = statistics_csv["ideal coverage"].to_numpy(dtype=float).flatten()
    for i, _ in enumerate(default_alphas):
        written_id_coverage = ideal_coverage[i*num_random:(i+1)*num_random-1]
        assert np.allclose(written_id_coverage, expected_coverages[i])
    # TODO: rotation statistics


def test_ordering():
    # TODO: figure out what's the issue
    """Assert that, ignoring randomness, the first N-1 points of ordered grid with length N are equal to ordered grid
    of length N-1"""
    for name in GRID_ALGORITHMS:
        try:
            for N in range(14, 284, 3):
                for addition in (1, 7):
                    grid_1 = build_grid(N + addition, name, use_saved=USE_SAVED).get_grid()
                    grid_2 = build_grid(N, name, use_saved=USE_SAVED).get_grid()
                    assert np.allclose(grid_1[:N], grid_2)
        except AssertionError:
            print(name)


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
    assert np.allclose(full_grid.b_rotations.rotations.as_matrix(), np.eye(3))

    full_grid = FullGrid(t_grid_name="[1, 2, 3]", o_grid_name="3", b_grid_name="4")
    assert full_grid.get_position_grid().shape == (3, 3, 3)
    assert full_grid.b_rotations.get_grid_z_as_array().shape == (4, 3)


def test_position_grid():
    num_rot = 14
    num_trans = 4  # keep this number unless you change t_grid_name
    fg = FullGrid(b_grid_name="zero", o_grid_name=f"ico_{num_rot}", t_grid_name="[0.1, 2, 2.5, 4]")
    ico_grid = build_grid(14, "ico", use_saved=USE_SAVED).get_grid()
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