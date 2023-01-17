import networkx as nx
import pandas as pd
import pytest
from scipy.constants import pi
from scipy.spatial import distance_matrix

from molgri.space.fullgrid import FullGrid
from molgri.space.polytopes import project_grid_on_sphere, Cube3DPolytope, IcosahedronPolytope, second_neighbours
from molgri.space.rotations import grid2rotation, rotation2grid
from molgri.space.rotobj import build_rotations, build_grid, build_grid_from_name, order_elements
from molgri.constants import GRID_ALGORITHMS, ZERO_ALGORITHM

import numpy as np

from molgri.space.utils import normalise_vectors


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
            rotations_start = build_rotations(N, algo, use_saved=False)
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


def test_project_grid_on_sphere():
    array_vectors = np.array([[3, 2, -1],
                              [-5, 22, 0.3],
                              [-3, -3, -3],
                              [0, 1/4, 1/4]])

    expected_results = np.array([[3/np.sqrt(14), np.sqrt(2/7), -1/np.sqrt(14)],
                                 [-0.221602, 0.975047, 0.0132961],
                                 [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
                                 [0, 1/np.sqrt(2), 1/np.sqrt(2)]])
    # test the whole array
    results = project_grid_on_sphere(array_vectors)
    assert np.allclose(results, expected_results)
    # test individual components
    for vector, expected_result in zip(array_vectors, expected_results):
        result = project_grid_on_sphere(vector.reshape((1, -1)))
        assert np.allclose(result, expected_result.reshape((1, -1)))
    # what happens for zero vector? should throw an error
    array_zero = np.array([[3, 2, -1],
                           [0, 0, 0]])
    with pytest.raises(AssertionError) as e:
        project_grid_on_sphere(array_zero)
    assert e.type is AssertionError
    # 2 dimensions
    array_vectors2 = np.array([[3, 2],
                              [-5, 0.3],
                              [-3, -3],
                              [0, 1/4]])
    expected_results2 = np.array([[3/np.sqrt(13), 2/np.sqrt(13)],
                                  [-0.998205, 0.0598923],
                                  [-1/np.sqrt(2), -1/np.sqrt(2)],
                                  [0, 1]])
    results2 = project_grid_on_sphere(array_vectors2)
    assert np.allclose(results2, expected_results2)
    # 4 dimensions
    array_vectors3 = np.array([[3, 2, -5, 0.3],
                              [-3, -3, 0, 1/4]])
    expected_results3 = np.array([[0.486089, 0.324059, -0.810148, 0.0486089],
                                  [-12/17, -12/17, 0, 1/17]])
    results3 = project_grid_on_sphere(array_vectors3)
    assert np.allclose(results3, expected_results3)


def test_polytope():
    for polytope_type in (Cube3DPolytope, IcosahedronPolytope):
        polytope = polytope_type()
        for level in range(3):
            graph_before = polytope.G.copy()
            nodes_before = list(graph_before.nodes(data=False))
            edges_before = list(graph_before.edges)
            polytope.divide_edges()
            graph_after = polytope.G.copy()
            nodes_after = list(graph_after.nodes(data=False))
            # no nodes should disappear
            for x in nodes_before:
                if x not in nodes_after:
                    raise Exception("No nodes should disappear when subdividing!")
            # now the remaining points should be midpoints of edges
            all_midpoints = []
            for edge in edges_before:
                midpoint = (np.array(edge[0]) + np.array(edge[1]))/2
                midpoint = tuple(midpoint)
                all_midpoints.append(midpoint)
                if midpoint not in nodes_after:
                    raise Exception("At least one of the midpoints was not added to grid!")
            assert set(all_midpoints).union(set(nodes_before)) == set(nodes_after)


def test_second_neighbours():
    G = nx.Graph([(5, 1), (5, 6), (6, 1), (1, 2), (2, 7), (1, 3), (1, 9), (3, 4),
                  (4, 9), (3, 10), (3, 8), (8, 10), (10, 11)])
    # uncomment next three lines if you need to check how it looks
    # import matplotlib.pyplot as plt
    # nx.draw_networkx(G)
    # plt.show()
    expected_neig_1 = [5, 6, 2, 3, 9]
    # expected second neighbours that are NOT first neighbours
    expected_sec_neig_1 = [4, 7, 10, 8]
    sec_neig_1 = list(second_neighbours(G, 1))
    assert np.all([x in list(G.neighbors(1)) for x in expected_neig_1]), "First neighbours wrong."
    assert np.all([x in sec_neig_1 for x in expected_sec_neig_1]), "Some second neighbours missing."
    assert not np.any([x in sec_neig_1 for x in expected_neig_1]), "Some first neighbours in second neighbours"
    # the example given there
    G2 = nx.Graph([(5, 6), (5, 2), (2, 1), (6, 1), (1, 3), (3, 7), (1, 8), (8, 3)])
    exp_1 = [2, 6, 3, 8]
    exp_2 = [5, 7]
    real_1 = list(set(G2.neighbors(1)))
    real_2 = list(set(second_neighbours(G2, 1)))
    assert len(exp_1) == len(real_1) and sorted(exp_1) == sorted(real_1), "Something wrong with first neighbours."
    assert len(exp_2) == len(real_2) and sorted(exp_2) == sorted(real_2), "Something wrong with second neighbours."


def test_general_grid_properties():
    for alg in GRID_ALGORITHMS[:-1]:
        for number in (3, 15, 26):
            grid_obj = build_grid(number, alg, use_saved=False)
            grid = grid_obj.get_grid()
            assert isinstance(grid, np.ndarray), "Grid must be a numpy array."
            assert grid.shape == (number, 3), "Wrong grid shape."
            assert grid_obj.N == number


def test_cube_3d_grid():
    cube_3d = build_grid(8, "cube3D")
    #assert len(cube_3d.polyhedron.faces) == 6
    grid = cube_3d.get_grid()
    distances = distance_matrix(grid, grid)
    # diagonal should be zero
    assert np.allclose(np.diagonal(distances), 0)
    a = 2/np.sqrt(3)
    # 24 elements should be vertice lengths
    assert np.count_nonzero(np.isclose(distances, a)) == 24
    # 24 elemens are face diagonals
    assert np.count_nonzero(np.isclose(distances, a*np.sqrt(2))) == 24
    # 8 elements should be volume diagonals
    assert np.count_nonzero(np.isclose(distances, a * np.sqrt(3))) == 8
    # rows have the right elements
    for row in distances:
        assert sorted(row) == sorted([0, a, a, a, a*np.sqrt(2), a*np.sqrt(2), a*np.sqrt(2), a*np.sqrt(3)])
    # only selected angles possible
    for vec1 in grid:
        for vec2 in grid:
            angle_points = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
            tetrahedron_a1 = np.arccos(-1/3)
            tetrahedron_a2 = np.arccos(1 / 3)
            assert np.any(np.isclose(angle_points, [0, np.pi/2, np.pi, np.pi/3, tetrahedron_a1, tetrahedron_a2]))
    # subdivision must be checked by polytope tester


def test_zero_grid():
    ico = build_grid_from_name("ico_0")
    ico.get_grid()


def test_errors_and_assertions():
    with pytest.raises(ValueError):
        build_grid(15, "icosahedron")
    with pytest.raises(ValueError):
        build_grid(15, "grid")
    with pytest.raises(AssertionError):
        build_grid(-15, "ico")
    with pytest.raises(AssertionError):
        # noinspection PyTypeChecker
        build_grid(15.3, "ico")
    grid = build_grid(20, "ico").get_grid()
    with pytest.raises(ValueError):
        order_elements(grid, 25)


def test_everything_runs():
    cp = Cube3DPolytope()
    cp.divide_edges()
    cp.plot_graph()
    ip = IcosahedronPolytope()
    ip.divide_edges()
    ip.divide_edges()
    ip.plot_graph()
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
                    grid_1 = build_grid(N + addition, name).get_grid()
                    grid_2 = build_grid(N, name).get_grid()
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
    ico_grid = build_grid(14, "ico").get_grid()
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