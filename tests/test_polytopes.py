import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

from molgri.space.polytopes import Cube3DPolytope, IcosahedronPolytope, second_neighbours, \
    Cube4DPolytope, third_neighbours, detect_all_cubes, detect_all_squares, PolyhedronFromG, \
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


def test_polytope():
    for polytope_type in ALL_POLYTOPE_TYPES:
        polytope = polytope_type()
        for level in range(2):
            graph_before = polytope.G.copy()
            nodes_before = polytope.get_node_coordinates() #list(graph_before.nodes(data=False))
            edges_before = list(graph_before.edges)
            polytope.divide_edges()
            nodes_after = polytope.get_node_coordinates()
            # no nodes should disappear
            for x in nodes_before:
                if x not in nodes_after:
                    raise Exception("No nodes should disappear when subdividing!")
            # now the remaining points should be midpoints of edges
            all_midpoints = []
            for edge in edges_before:
                midpoint = np.average(polytope.get_node_coordinates(for_nodes=edge[:2]), axis=0)
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


def test_everything_runs():
    # test all
    for polytope_type in ALL_POLYTOPE_TYPES:
        pol = polytope_type()
        pol.divide_edges()
        all_row_norms_similar(pol.get_projection_coordinates())


def test_level0():
    # all
    for poly_type in ALL_POLYTOPE_TYPES:
        pol = poly_type()
        points = pol.get_node_coordinates()
        projections = pol.get_projection_coordinates()
        # all level attributes must be 0
        for n in pol.G.nodes(data=True):
            assert n[1]["level"] == 0, "All points should be level 0 right after creation!"
        # number of points at level 0
        points_at_0 = len(projections)
        # afterwards, projections always of length 1
        all_row_norms_equal_k(projections, 1)
        pol.divide_edges()
        projections_1 = pol.get_projection_coordinates()
        all_row_norms_equal_k(projections_1, 1)
        # the ordering of projections does not change
        projections_2 = pol.get_projection_coordinates()
        assert np.allclose(projections_1, projections_2)
        # the first points_at_0 projections and coordinates should still be the same
        assert np.allclose(pol.get_node_coordinates()[:points_at_0], points)
        assert np.allclose(projections_2[:points_at_0], projections)


def test_ico_polytope():
    #  icosahedron
    ico = IcosahedronPolytope()
    assert ico.G.number_of_nodes() == 12, "Icosahedron should have 12 nodes"
    assert len(ico.get_node_coordinates()) == 12, "Icosahedron should have 12 nodes"
    assert len(ico.get_projection_coordinates()) == 12, "Icosahedron should have 12 nodes"
    assert ico.G.number_of_edges() == 30, "Icosahedron should have 30 edges"


    # after one division
    ico.divide_edges()
    # for each edge new point
    all_rows_unique(ico.get_projection_coordinates())
    assert ico.G.number_of_nodes() == 12 + 30, "1st division: icosahedron should have 42 nodes"
    # each of 20 faces has 6 shared edges
    assert ico.G.number_of_edges() == 20 * 3, "1st division: icosahedron should have 60 edges"
    # after two divisions
    ico.divide_edges()
    # for each edge new point
    all_rows_unique(ico.get_projection_coordinates())
    # fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    # ico.plot_points(ax, select_faces={1, 2, 3})
    # ico.plot_edges(ax, select_faces={1, 2, 3})
    # plt.show()
    assert ico.G.number_of_nodes() == 42 + 120, "2nd division: icosahedron should have 162 nodes"
    # each of 20 faces: count edges and shared edges
    assert ico.G.number_of_edges() == 240, "2nd division: icosahedron should have 240 edges"


def test_cube3D_polytope():
    # cube3D
    cub = Cube3DPolytope()
    # each of 6 faces has one whole point and 4 * 1/3 points (shared by three sides)
    assert cub.G.number_of_nodes() == 14, "Cube 3D should have 14 nodes"
    # those points are unique
    all_rows_unique(cub.get_projection_coordinates())
    # each face has the occupancy of 5 nodes
    # normal cube has 12 edges, but we add 6*4 face diagonals
    assert cub.G.number_of_edges() == 12 + 24, "Cube should have 36 edges"
    # after one division
    cub.divide_edges()
    # for each edge new point
    all_rows_unique(cub.get_projection_coordinates())
    # each of 6 faces has 5 whole points + 4 * 1/3 + 4 * 1/2
    assert cub.G.number_of_nodes() == 50, "1st division: cube should have 50 nodes"
    # each of previous edges simply cut in half
    assert cub.G.number_of_edges() == 72, "1st division: cube should have 72 edges"
    # after two divisions
    cub.divide_edges()
    # fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    # cub.plot_points(ax, select_faces={1, 2, 3})
    # cub.plot_edges(ax, select_faces={1, 2, 3})
    # plt.show()
    all_rows_unique(cub.get_projection_coordinates())
    # each of 6 faces has 25 whole points + 4 * 1/3 + 12 * 1/2
    assert cub.G.number_of_nodes() == 194, "2nd division: cube should have 194 nodes"
    assert cub.G.number_of_edges() == 288, "2nd division: cube should have 288 edges"
    # after 3 divisions
    cub.divide_edges()
    # 194 old points + 576 for each edge sub-divided = 770
    assert cub.G.number_of_nodes() == 770, "3rd division: cube should have 770 nodes"
    assert cub.G.number_of_edges() == 4*288, "3rd division: cube should have 1152 edges"


def test_cube4D_polytope():
    cub = Cube4DPolytope()
    assert cub.G.number_of_nodes() == 16, "Hypercube should have 16 nodes"
    # categories of nodes
    node_categories = cub._get_count_of_point_categories()
    assert node_categories == [16], "At level 0 cube 4D has point categories [16]"
    # those points are unique
    all_rows_unique(cub.get_projection_coordinates())
    assert cub.G.number_of_edges() == 112, "Hypercube should have 112 edges"
    # categories of edges
    edge_categories = cub._get_count_of_edge_categories()
    assert np.all(edge_categories == [32, 48, 32]), "At level 0 cube 4D has edge categories [32, 48, 32]"

    # after one division
    cub.divide_edges()
    # why 80?  One point per: vertex, edge, face, cell, 16 + 32 + 24 + 8 = 80
    assert cub.G.number_of_nodes() == 80, "1st division: hypercube should have 80 nodes"
    node_categories = cub._get_count_of_point_categories()
    assert np.all(node_categories == [16, 32, 24, 8]), "At level 1 cube 4D has point categories [16, 32, 24, 8]"
    n_edges = cub.G.number_of_edges()
    print(cub._get_count_of_edge_categories())
    assert n_edges == 848, f"1st division: hypercube should have 848 edges, not {n_edges}"
    edge_categories = cub._get_count_of_edge_categories()
    assert np.all(edge_categories == [208, 384, 256]), "At level 1 cube 4D has edge categories [208, 384, 256]"

    # after two divisions
    cub.divide_edges()
    assert cub.G.number_of_nodes() == 544, f"2nd division: hypercube should have 544 nodes, not {cub.G.number_of_nodes()}"
    assert cub.G.number_of_edges() == 6304, f"2nd division: hypercube should have 1568 edges, not {cub.G.number_of_edges()}"
    node_categories = cub._get_count_of_point_categories()
    assert np.all(node_categories == [16, 64, 32, 96, 96, 24, 64, 96, 48, 8]), "At level 2 cube 4D has point categories [16, 64, 32, 96, 96, 24, 64, 96, 48, 8]"
    edge_categories = cub._get_count_of_edge_categories()
    assert np.all(edge_categories == [1184, 3072, 2048]), "At level 2 cube 4D has edge categories [1184, 3072, 2048]"


def test_sorting():
    my_G = example_cube_graph()
    polyh = PolyhedronFromG(my_G)
    side_len = 1
    np.random.seed(1)
    # N < number of points, N == number of points, N > number of points
    for N1 in (4, 8):
        points_before = polyh.get_node_coordinates()
        projections_before = polyh.get_projection_coordinates()
        sorted_points = polyh.get_N_ordered_points(N1, projections=False)
        sorted_projections = polyh.get_N_ordered_points(N1, projections=True)
        points_after = polyh.get_node_coordinates()
        projections_after = polyh.get_projection_coordinates()
        # first assert it does not mess up the representation of all points
        assert np.allclose(points_before, points_after)
        assert np.allclose(projections_before, projections_after)
        # secondly assert the right shape
        assert sorted_points.shape == (N1, 3)
        assert sorted_projections.shape == (N1, 3)
        # assert that the sorted points come exactly from the list of points
        assert np.all([point in points_before for point in sorted_points])
        assert np.all([point in projections_before for point in sorted_projections])



def test_detect_square_and_cubes():
    my_G = example_cube_graph()
    polyh = PolyhedronFromG(my_G)
    # you should detect 6 square faces in a 3D cube consisting of only 8 vertices
    assert len(detect_all_squares(my_G)) == 6
    # and you should detect exactly one cube
    assert len(detect_all_cubes(my_G)) == 1
    # you should also be able to add those points to the polyhedron
    points_before = polyh.G.number_of_nodes()
    polyh._add_square_diagonal_nodes()
    points_after = polyh.G.number_of_nodes()
    assert points_after - points_before == 6


def test_edge_attributes():
    # ico
    ico = IcosahedronPolytope()
    side_len = 1 / np.sin(2 * pi / 5)
    arch_len = np.arccos(1-side_len**2/2)
    # after subdivision:
    for i in range(3):
        edges = ico.G.edges(data=True)
        for n1, n2, data in edges:
            assert np.isclose(data["p_dist"], side_len)
            #assert np.isclose(data["length"], arch_len)
        ico.divide_edges()
        side_len = side_len / 2
    # cube3D
    cube3D = Cube3DPolytope()
    side_len = 2 * np.sqrt(1/3)
    # after subdivision:
    for i in range(3):
        edges = cube3D.G.edges(data=True)
        for n1, n2, data in edges:
            assert np.isclose(data["p_dist"], side_len) or np.isclose(data["p_dist"], np.sqrt(2)*side_len/2)
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
    for i in range(3):
        edges = cube4D.G.edges(data=True)
        for n1, n2, data in edges:
            assert np.isclose(data["p_dist"], side_len) or np.isclose(data["p_dist"], np.sqrt(2)*side_len/2) or \
                   np.isclose(data["p_dist"], np.sqrt(3)*side_len/2) or np.isclose(data["p_dist"], side_len/2)
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
    # test_polytope()
    # test_second_neighbours()
    # test_third_neighbours()
    # test_everything_runs()
    # test_level0()
    # test_ico_polytope()
    # test_cube3D_polytope()
    test_cube4D_polytope()
    # test_sorting()
    # test_detect_square_and_cubes()
    # test_edge_attributes()
    # test_remove_and_reconnect()