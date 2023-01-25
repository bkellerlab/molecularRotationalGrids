import networkx as nx
import numpy as np
import pytest
import matplotlib.pyplot as plt

from molgri.space.polytopes import Cube3DPolytope, IcosahedronPolytope, second_neighbours, project_grid_on_sphere,\
    Cube4DPolytope
from molgri.assertions import all_row_norms_similar, all_row_norms_equal_k

ALL_POLYTOPE_TYPES = (Cube3DPolytope, IcosahedronPolytope, Cube4DPolytope)
ALL_POLYHEDRON_TYPES = (Cube3DPolytope, IcosahedronPolytope)


def test_polytope():
    for polytope_type in ALL_POLYHEDRON_TYPES:
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


def test_everything_runs():
    # test all
    for polytope_type in ALL_POLYTOPE_TYPES:
        pol = polytope_type()
        pol.plot_graph()
        pol.divide_edges()
        pol.plot_graph(with_labels=False)
        all_row_norms_similar(pol.get_projection_coordinates())
    # test only 3D
    for polytope_type in ALL_POLYHEDRON_TYPES:
        pol = polytope_type()
        pol.divide_edges()
        pol.divide_edges()
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        pol.plot_points(ax, select_faces={3, 4}, projection=True)
        pol.plot_points(ax, select_faces=None, projection=False)
        pol.plot_edges(ax, select_faces={8})


def test_level0():
    # all
    for poly_type in ALL_POLYTOPE_TYPES:
        pol = poly_type()
        points = pol.get_node_coordinates()
        # upon creation, all points of length 1
        all_row_norms_equal_k(points, 1)
        projections = pol.get_projection_coordinates()
        # afterwards, projections always of length 1
        all_row_norms_equal_k(projections, 1)
        pol.divide_edges()
        projections_1 = pol.get_projection_coordinates()
        all_row_norms_equal_k(projections_1, 1)
        # the ordering of projections does not change
        pol.get_projection_coordinates()
        projections_2 = pol.get_projection_coordinates()
        assert np.allclose(projections_1, projections_2)
    #  icosahedron
    ico = IcosahedronPolytope()
    assert ico.G.number_of_nodes() == 12, "Icosahedron should have 12 nodes"
    assert ico.G.number_of_edges() == 30, "Icosahedron should have 30 edges"
    # after one division
    ico.divide_edges()
    # for each edge new point / 2
    assert ico.G.number_of_nodes() == 12 + 15, "1st division: icosahedron should have 27 nodes"
    # each point has 5 connections / 2
    assert ico.G.number_of_edges() == 27 * 5 // 2, "1st division: icosahedron should have 67 edges"



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


if __name__ == "__main__":

    cp = Cube3DPolytope()
    #cp.plot_graph(with_labels=False)

    plt.show()