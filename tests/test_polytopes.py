import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

from molgri.space.polytopes import Cube3DPolytope, IcosahedronPolytope, second_neighbours, \
    Cube4DPolytope, third_neighbours, PolyhedronFromG, \
    remove_and_reconnect
from molgri.assertions import all_row_norms_similar, all_row_norms_equal_k, all_rows_unique
from molgri.space.utils import normalise_vectors, dist_on_sphere

ALL_POLYTOPE_TYPES = (Cube3DPolytope, IcosahedronPolytope, Cube4DPolytope)
ALL_POLYHEDRON_TYPES = (Cube3DPolytope, IcosahedronPolytope)


def example_cube_graph() -> nx.Graph:
    """
    Simple cube graph with 8 nodes numbered 1-8, each being connected to exactly 3 other nodes.
    """
    G = nx.Graph([(1, 2), (1, 4), (1, 5), (2, 3), (3, 4), (2, 6), (3, 7), (4, 8), (5, 6), (6, 7), (7, 8), (5, 8)])

    # make the graph more like a normal polyhedron
    previous = list(range(1, 9))
    coords = [(1/2, -1/2, 1/2), (-1/2, -1/2, 1/2), (-1/2, 1/2, 1/2), (1/2, 1/2, 1/2), (1/2, -1/2, -1/2),
              (-1/2, -1/2, -1/2), (-1/2, 1/2, -1/2), (1/2, 1/2, -1/2)]
    # add level and projection
    nx.set_node_attributes(G, {p: c for p, c in zip(previous, coords)}, name="polytope_point")
    nx.set_node_attributes(G, 0, name="level")
    nx.set_node_attributes(G, set(), name="face")
    nx.set_node_attributes(G, {node: G.nodes[node]["polytope_point"]/np.linalg.norm(G.nodes[node]["polytope_point"])
                               for node in G.nodes}, name="projection")
    dist = {(n1, n2): dist_on_sphere(G.nodes[n1]["projection"], G.nodes[n2]["projection"])[0] for n1, n2 in G.edges}
    nx.set_edge_attributes(G, dist, name="length")
    side_len = 1
    nx.set_edge_attributes(G, side_len, name="p_dist")
    return G


def test_getter():
    """
    Assert that the method get_nodes
     1) consistently returns the same values when reconstructing the object from scratch
     2) returns the values in the order of CI
     3) CI is a unique, increasing integer
    """
    for polytope_type in ALL_POLYTOPE_TYPES:
        polytope = polytope_type()
        polytope.divide_edges()
        N_nodes_1 = polytope.G.number_of_nodes()
        # test getting CI - should return a list of indices in correct ordering (from 0 to num of nodes - 1)
        all_ci = polytope._get_attributes_array_sorted_by_index("central_index")
        assert np.allclose(all_ci, list(range(polytope.G.number_of_nodes())))
        # test getting nodes and projections for two separately constructed objects
        polytope_2 = polytope_type()
        polytope_2.divide_edges()
        polytope_2.divide_edges()

        # all nodes/projections give same order and up to N nodes/projections too
        for N in (None, 16):
            for projection in (True, False):
                if N is None:
                    up_to = N_nodes_1
                else:
                    up_to = N
                all_nodes_1 = polytope.get_nodes(N=N, projection=projection)
                all_nodes_2 = polytope_2.get_nodes(N=N, projection=projection)
                assert np.allclose(all_nodes_1[:up_to], all_nodes_2[:up_to])

        # getters obtain results in order of the CI
        my_N = N_nodes_1 - 2
        my_points = polytope.get_nodes(N=my_N, projection=False)
        for i, point in enumerate(my_points):
            assert polytope.G.nodes[tuple(point)]["central_index"] == i
            assert polytope_2.G.nodes[tuple(point)]["central_index"] == i


def test_basics():
    """
    Tests that:
    1) all polytopes can be created
    2) no nodes disappear when subdividing
    3) midpoints of all edges are added to the polytope during division
    4) projected points all have the same norm and it is 1
    5) upon creation, all points have level 0
    """
    for polytope_type in ALL_POLYTOPE_TYPES:
        polytope = polytope_type()
        for n in polytope.G.nodes(data=True):
            assert n[1]["level"] == 0, "All points should be level 0 right after creation!"
        for level in range(2):
            graph_before = polytope.G.copy()
            nodes_before = polytope.get_nodes()
            edges_before = list(graph_before.edges)
            all_row_norms_equal_k(polytope.get_nodes(projection=True), 1)
            polytope.divide_edges()
            nodes_after = polytope.get_nodes()
            # no nodes should disappear
            for x in nodes_before:
                if x not in nodes_after:
                    raise Exception("No nodes should disappear when subdividing!")
            # now the remaining points should be midpoints of edges
            all_midpoints = []
            for edge in edges_before:
                midpoint = np.average(np.array(edge[:2]), axis=0)
                all_midpoints.append(midpoint)
                if midpoint not in nodes_after:
                    raise Exception(f"At least one of the midpoints was not added to grid for {polytope_type}!")


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
    real_1 = list(G2.neighbors(1))
    real_2 = list(second_neighbours(G2, 1))
    assert len(exp_1) == len(real_1) and sorted(exp_1) == sorted(real_1), "Something wrong with first neighbours."
    assert len(exp_2) == len(real_2) and sorted(exp_2) == sorted(real_2), "Something wrong with second neighbours."


def test_third_neighbours():
    # should be the example in the description of the function
    G = nx.Graph([(5, 2), (5, 6), (6, 1), (1, 2), (1, 3), (3, 7), (7, 9),
                  (1, 11), (11, 8), (8, 10), (10, 3)])
    # uncomment next three lines if you need to check how it looks
    # import matplotlib.pyplot as plt
    # nx.draw_networkx(G)
    # plt.show()

    # First neighbours of 1: 2, 6, 3, 11
    assert sorted(list(G.neighbors(1))) == sorted([2, 6, 3, 11])
    # Second neighbours of 1: 5, 8, 10, 7
    assert sorted(second_neighbours(G, 1)) == sorted([5, 8, 10, 7])
    # Third neighbours of 1: 9
    assert sorted(third_neighbours(G, 1)) == sorted([9])
    # third neighbours of 7: 2, 6, 11, 8
    assert sorted(third_neighbours(G, 7)) == sorted([2, 6, 11, 8])


def test_ico_polytope():
    """
    Tests that:
    1) the number of nodes and edges of IcosahedronPolytope are correct at levels 0, 1 and 2
    2) the generated points are all unique
    """

    expected_num_of_points = [12, 42, 162]
    expected_num_of_edges = [30, 60, 240]

    ico = IcosahedronPolytope()
    for i in range(3):
        if i != 0:
            ico.divide_edges()
        assert ico.G.number_of_nodes() == expected_num_of_points[i], f"At level {i} cube 3D should have {expected_num_of_points[i]} nodes"
        # those points are unique
        all_rows_unique(ico.get_nodes(projection=True))
        assert ico.G.number_of_edges() == expected_num_of_edges[i], f"At level {i} cube 3D should have {expected_num_of_edges[i]} edges "


def test_cube3D_polytope():
    """
    This test asserts that:
    1) the number of nodes and edges of Cube3DPolytope are correct at levels 0, 1 and 2
    2) The categories of edges (straight, diagonal) are also correct
    3) the generated points are all unique
    """
    expected_num_of_points = [8, 26, 98]
    expected_num_of_edges = [24, 96, 384]
    expected_categories_of_edges = [[12, 12], [48, 48], [192, 192]]

    # cube3D
    cub = Cube3DPolytope()
    for i in range(3):
        if i != 0:
            cub.divide_edges()
        assert cub.G.number_of_nodes() == expected_num_of_points[i], f"At level {i} cube 3D should have {expected_num_of_points[i]} nodes"
        # those points are unique
        all_rows_unique(cub.get_nodes(projection=True))
        assert cub.G.number_of_edges() == expected_num_of_edges[i], f"At level {i} cube 3D should have {expected_num_of_edges[i]} edges "
        edge_categories = cub._get_count_of_edge_categories()
        assert np.all(edge_categories == expected_categories_of_edges[i]), f"At level {i} cube 3D has edge categories {expected_categories_of_edges[i]}"


def test_cube4D_polytope():
    """
    This test asserts that:
    1) the number of nodes and edges of Cube3DPolytope are correct at levels 0, 1 and 2
    2) The categories of nodes (distance to hyperspher) are correct
    3) The categories of edges (straight, diagonal) are also correct
    3) the generated points are all unique
    """

    expected_num_of_points = [16, 80, 544]
    expected_num_of_edges = [112, 848, 6688]
    expected_categories_of_nodes = [[16], [16, 32, 24, 8], [16, 64, 32, 96, 96, 24, 64, 96, 48, 8]]
    expected_categories_of_edges = [[32, 48, 32], [208, 384, 256], [1568, 3072, 2048]]


    cub = Cube4DPolytope()
    for i in range(3):
        if i != 0:
            cub.divide_edges()
        assert cub.G.number_of_nodes() == expected_num_of_points[i], f"At level {i} cube 4D should have " \
                                                                     f"{expected_num_of_points[i]} nodes"
        # those points are unique
        all_rows_unique(cub.get_nodes(projection=True))
        assert cub.G.number_of_edges() == expected_num_of_edges[i], f"At level {i} cube 4D should have " \
                                                                    f"{expected_num_of_edges[i]} edges "
        node_categories = cub._get_count_of_point_categories()
        assert np.all(node_categories == expected_categories_of_nodes[i]), f"At level {i} cube 4D has edge categories {expected_categories_of_edges[i]}"
        edge_categories = cub._get_count_of_edge_categories()
        assert np.all(edge_categories == expected_categories_of_edges[i]), f"At level {i} cube 4D has edge categories {expected_categories_of_edges[i]}"


def test_edge_attributes():
    """
    This function tests that:
    1) for all polytopes, the distances on polytopes are the expected lengths (for straight, square diag, cube diag)
    """
    # ico
    ico = IcosahedronPolytope()
    side_len = 1 / np.sin(2 * pi / 5)
    # after subdivision:
    for i in range(3):
        edges = ico.G.edges(data=True)
        for n1, n2, data in edges:
            assert np.isclose(data["p_dist"], side_len)
        ico.divide_edges()
        side_len = side_len / 2

    # cube3D
    cube3D = Cube3DPolytope()
    side_len = 2 * np.sqrt(1/3)
    # after subdivision:
    for i in range(3):
        edges = cube3D.G.edges(data=True)
        for n1, n2, data in edges:
            assert np.isclose(data["p_dist"], side_len) or np.isclose(data["p_dist"], np.sqrt(2)*side_len)
        # fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        # cube3D.plot_points(ax, color_by="index")
        # cube3D.plot_edges(ax, label="p_dist")
        # plt.show()
        cube3D.divide_edges()
        side_len = side_len / 2
    # cube4D
    cube4D = Cube4DPolytope()
    side_len = 2 * np.sqrt(1/4)
    # after subdivision:
    for i in range(2):
        edges = cube4D.G.edges(data=True)
        for n1, n2, data in edges:
            assert np.isclose(data["p_dist"], side_len) or \
                   np.isclose(data["p_dist"], np.sqrt(3)*side_len) or np.isclose(data["p_dist"], np.sqrt(2)*side_len)
        cube4D.divide_edges()
        side_len = side_len / 2


def test_remove_and_reconnect():
    # if I remove point 2, I want connection betwen 1 and 3
    my_G = nx.Graph([(1, 2), (2, 3)])
    nx.set_edge_attributes(my_G, 1, name="p_dist")
    assert (1, 2) in my_G.edges and (2, 3) in my_G.edges
    remove_and_reconnect(my_G, 2)
    assert (1, 3) in my_G.edges and len(my_G.edges) == 1


if __name__ == "__main__":
    test_getter()
    test_basics()
    test_second_neighbours()
    test_third_neighbours()
    test_ico_polytope()
    test_cube3D_polytope()
    test_cube4D_polytope()
    test_edge_attributes()
    test_remove_and_reconnect()