from typing import Generator, Callable, Optional

from numpy.typing import ArrayLike, NDArray
import numpy as np
import networkx as nx
from scipy.sparse import coo_array
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd

from molgri.constants import DIM_SQUARE


def from_edges_path_to_simple_path(edges_path: list) -> list:
    """
    Convert path format: [(i, k1), (k1, k2), ...(km, j)] -> [i, k1, k2, ... km, j]

    Args:
        edges_path (list): a list of tuples, each tuple represents an edge travelled in a path

    Returns:
        a list of integers, each integer represents a node travelled in a path, including the first and the last one

    """
    # first element of each tuple + last element of last tuple
    simple_path = [el[0] for el in edges_path]
    simple_path.append(edges_path[-1][-1])
    return simple_path

class Maze:

    """
    Object used to search for paths in adjacency matrix weighted by cell energies and similar.
    """

    def __init__(self, adjacency_matrix):
        """
        A maze consists of connected cells, each cell can have properties like Boltzmann weight or energy.

        Args:
            adjacency_matrix (numpy matrix or scipy sparse matrix): a NxN binary matrix where entry (i,j)=True means
            that these two cells are connected (neighbours)
        """
        self.maze = nx.Graph(adjacency_matrix)
        self.adjacency_matrix = coo_array(adjacency_matrix)
        self.energy_name: Optional[str] = None
        self.energy_array: Optional[NDArray] = None

    def add_energy_attribute(self, energy_name: str, energy_values: ArrayLike) -> None:
        """
        Add the energy information to every node of the maze.

        Args:
            energy_name (str): provide a name (preferably with units!) to refer to the energy of the cells
            energy_values (ArrayLike): provide an ordered list of energy values for each node
        """
        # register energy name for use in other functions
        self.energy_name = energy_name
        self.energy_array = np.array(energy_values)

        assert len(energy_values) == self.maze.number_of_nodes()

        nx.set_node_attributes(self.maze, dict(enumerate(energy_values)), name=energy_name)
        # edge attributes - differences in energy
        diff_energies = self.energy_array[self.adjacency_matrix.row] - self.energy_array[self.adjacency_matrix.col]
        diff_energies_coo = self.adjacency_matrix.copy()
        diff_energies_coo.data = diff_energies

        edge_delta_E = {(i, j): max([float(d), 0.0]) for i, j, d in zip(diff_energies_coo.row,
                                                     diff_energies_coo.col,
                                                     diff_energies_coo.data)}
        nx.set_edge_attributes(self.maze, edge_delta_E, name="ΔE")

    def get_energies(self, indices: list) -> NDArray:
        """
        Get energies list for indices list (ordered).

        Args:
            indices (list): indices of nodes for which we want the energies

        Returns:
            a filtered array of energies featuring only selected indices
        """
        return np.take(self.energy_array, indices)

    def all_paths_ij(self, i: int, j: int, cutoff=None) -> Generator:
        """
        Find a collection of all paths (with no node repeated) between cells with indices i and j.

        Args:
            i (int): start index (of adjacency matrix)
            j (int): end index (of adjacency matrix)

        Returns:
            a generator, each generated element is of form [(i, k1), (k1, k2), (k2, k3), ... (km, j)] for a
            connecting path i, k1, k2, k3, ... km, j
        """
        return nx.all_simple_edge_paths(self.maze, i, j, cutoff=cutoff)

    def _largest_barrier_metric(self, simple_path: ArrayLike) -> float:
        """
        Get the maximal barrier that is found along a path from energies of individual cells. This equals the 
        difference between the highest energy encountered and the start energy.
        
        Args:
           simple_path (ArrayLike): a sequence of integers indicating the nodes visited during the path

        Returns:
            A value of max energy difference >= 0. The smaller the energy difference, the more favourable this path is.
        """""
        energies_along_path = self.get_energies(simple_path)
        energy_start = energies_along_path[0]
        return np.max(energies_along_path) - energy_start

    def _opt_weighted_path_ij(self, i: int, j: int, weight_function: Callable = None):
        pass

    def _weighted_paths_ij(self, i: int, j: int, weight_function: Callable = None, cutoff=None) -> pd.DataFrame:
        """
        Among all possible paths between i and j find the value of the weight function that depends only on a list of
        nodes encountered along the path (in correct order).

        Args:
            i (int): index of start cell
            j (int): index of end cell
            weight_function (Callable): a function of path indices that returns a scalar, the lower the scalar,
            the better the path

        Returns:
            dataframe, each row contains the information (simple_path, weight_of_this_path)
        """
        data = []
        for path in self.all_paths_ij(i, j, cutoff=cutoff):
            simple_path = from_edges_path_to_simple_path(path)
            path_weight = weight_function(simple_path)
            data.append([tuple(simple_path), path_weight])
        df = pd.DataFrame(data, columns=["Path", "Path weight"])
        return df

    def max_barrier_paths_i_lenk(self, i: int, len_k: int) -> pd.DataFrame:
        print(nx.single_source_shortest_path(self.maze, i, cutoff=len_k))

    def max_barrier_paths_ij(self, i: int, j: int, cutoff=None) -> pd.DataFrame:
        """
        For each path between i and j find the maximal energy barrier on this path.

        Args:
            i (int): index of start cell
            j (int): index of end cell

        Returns:
            dataframe, each row contains the information (simple_path, highest_barrier_of_this_path)

        """
        if self.energy_name is None:
            raise ValueError("Cannot find a path based on energy if no energy information provided.")
        return self._weighted_paths_ij(i, j, weight_function=self._largest_barrier_metric, cutoff=cutoff)

    def plot_profile_along_path_df(self, path_df: pd.DataFrame, max_plot_rows: int = 6) -> (Figure, Axes):
        """
        Sorted by weight, plot the path elements of this dataframe using self.plot_profile_along_path.

        Args:
            path_df (df): A dataframe with columns "Path", "Path weight"

        Returns:
            a multiplot of paths, the one with smallest weight on top
        """
        # sort dataframe by weight
        path_df.sort_values(by="Path weight", inplace=True)

        num_rows = np.min([max_plot_rows, len(path_df)])
        fig, ax = plt.subplots(num_rows, 1, sharey=True, sharex=True, figsize=(DIM_SQUARE[0], num_rows*DIM_SQUARE[1]))

        for i in range(num_rows):
            row = path_df.iloc[i]
            path = row["Path"]
            weight = row["Path weight"]
            self.plot_profile_along_path(path, fig, ax[i])
            ax_r = ax[i].twinx()
            ax_r.set_yticks([])
            ax_r.set_ylabel(f'ΔE={weight:.2f}', color='r')

        fig.supxlabel(ax[0].get_xlabel())
        fig.supylabel(ax[0].get_ylabel())

        for i in range(num_rows):
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")

        plt.tight_layout()
        return fig, ax

    def plot_profile_along_path(self, simple_path: list, fig: Figure, ax: Axes) -> (Figure, Axes):
        """
        Given a list of node indices, find their corresponding energies and plot them in a schematic "barrier profile
        along reaction coordinate" way.

        Args:
            simple_path (list): a sequence of integers [i, k1, k2, k3, ... km, j) representing a path between i and j
            through nodes k1 ... km - order is important!
        """
        energies_along_path = self.get_energies(simple_path)

        num_states = len(energies_along_path)
        x_max = 1
        x_min = 0

        len_state = (x_max - x_min) / num_states
        # a small break between the horizontal lines
        space_between = 0.3*len_state
        len_state -= space_between

        for i, energy in enumerate(energies_along_path):
            # plot horizontal lines
            end_hline = i*(len_state+space_between)+len_state
            ax.hlines(energy, i*(len_state+space_between), end_hline, color="black",
                      linewidth=1, alpha=1)

            # labels of states
            mid_hline = 0.5 * ((i*(len_state+space_between)) + (i*(len_state+space_between)+len_state))

            y_offset = 0.02
            ax.text(mid_hline, energy+y_offset, f"{simple_path[i]}")

            # plot diagonal lines
            if i+1 < len(energies_along_path):
                next_energy = energies_along_path[i+1]
                start_next_hline = end_hline + space_between
                ax.plot((end_hline, start_next_hline), (energy, next_energy), linestyle="--", color="black",
                        linewidth=0.5, alpha=1)

        # x axis
        ax.xaxis.set_ticks([])
        ax.set_xlabel("Reaction coordinate")

        # y axis
        ax.set_ylabel(self.energy_name)

        # TODO: have attribute length in edges and adjust the plot so that it represents how far away states are
        return fig, ax


    def plot_maze_index_energy(self, only_nodes: list = None) -> None:
        """
        Create a visualization of a maze featuring:
        - a label of cell index and energy
        - node colored according to its energy
        - possibility to plot a subset of nodes


        Args:
            only_nodes (list): only plot the nodes with specified indices
        """

        dict_node_energy = nx.get_node_attributes(self.maze, self.energy_name)

        if self.energy_name is None:
            nx.draw_networkx(self.maze, nodelist=only_nodes, node_size=700)
        else:
            # text on the node
            full_labels = {key: f"{item[0]}\nE={item[1]}" for key, item in zip(dict_node_energy.keys(),
                                                                               dict_node_energy.items())}

            node_colors = list(nx.get_node_attributes(self.maze, self.energy_name).values())

            # if only a subset of nodes should be drawn we need to provide the right colors
            if only_nodes is not None:
                node_colors = [node_colors[i] for i in sorted(only_nodes)]

            nx.draw_networkx(self.maze, labels=full_labels, node_color=node_colors, cmap="coolwarm", nodelist=only_nodes,
                             node_size=700)

        plt.show()
