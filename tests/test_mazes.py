import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
import matplotlib.pyplot as plt


from molgri.space.mazes import Maze, from_edges_path_to_simple_path

def get_7cell_maze():
    return np.array([[0, 1, 0, 0, 0, 0, 1],
                       [1, 0, 1, 1, 0, 0, 1],
                       [0, 1, 0, 1, 0, 0, 0],
                       [0, 1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 1, 0, 1],
                       [1, 1, 0, 0, 0, 1, 0]
                       ], dtype=bool)


def get_7cell_energies() -> list:
    return [0.2, -0.28, 0.1, 0.3, -0.27, -0.7, 0.7]


def test_nonweighted_maze():
    my_adj = get_7cell_maze()
    correct_edges = {(0, 1), (0, 6), (1, 2), (1, 3), (1, 6), (2, 3), (3, 4), (4, 5), (5, 6)}

    my_maze = Maze(my_adj)

    # assert all the right connections (and none extra)
    assert set(my_maze.maze.edges) == correct_edges

    # repeat test with sparse matrix
    my_sparse_adj = coo_matrix(my_adj)
    my_sparse_maze = Maze(my_sparse_adj)

    assert set(my_sparse_maze.maze.edges) == correct_edges

def test_weighted_maze():
    my_adj = get_7cell_maze()
    my_energies = get_7cell_energies()

    my_maze = Maze(my_adj)
    my_maze.add_energy_attribute("energy", my_energies)

    my_maze.plot_maze_index_energy()

    one_simple_path = [2, 1, 6, 5, 4]

    # fig, ax = plt.subplots(1, 1)
    # fig, ax = my_maze.plot_profile_along_path(one_simple_path, fig, ax)
    # plt.show()


def test_paths():
    my_adj = get_7cell_maze()
    my_maze = Maze(my_adj)
    my_maze.add_energy_attribute("energy", get_7cell_energies())

    all_expected_paths = {(2, 1, 0),
                          (2, 1, 3, 4, 5, 6, 0),
                          (2, 1, 6, 0),
                          (2, 3, 1, 0),
                          (2, 3, 1, 6, 0),
                          (2, 3, 4, 5, 6, 0),
                          (2, 3, 4, 5, 6, 1, 0)}

    # all paths between 2 and 0
    all_simple_paths = set()
    for path in my_maze.all_paths_ij(2, 0):
        all_simple_paths.add(tuple(from_edges_path_to_simple_path(path)))

    assert all_simple_paths == all_expected_paths

    # among these paths the one with lowest barrier
    df_2_0 = my_maze.max_barrier_paths_ij(2, 0)
    my_maze.max_barrier_paths_i_lenk(2, 4)
    # TODO: plot all these barriers above each other
    fig, ax = my_maze.plot_profile_along_path_df(df_2_0)
    plt.show()




if __name__ == "__main__":
    test_nonweighted_maze()
    test_paths()
    test_weighted_maze()