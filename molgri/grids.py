from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from scipy.constants import pi, golden
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import pandas as pd

from .analysis import random_sphere_points, unit_dist_on_sphere, random_axes_count_points
from .constants import DEFAULT_SEED, SIX_METHOD_NAMES, UNIQUE_TOL, ENDING_GRID_FILES
from .parsers import NameParser
from .paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_STAT
from .rotations import grid2quaternion, grid2euler, quaternion2grid, euler2grid
from .wrappers import time_method


class Polytope(ABC):

    def __init__(self):
        """
        A polytope is a 3-dim object consisting of a set of vertices and connections between them (edges) saved
        in self.G property.
        """
        self.G = nx.Graph()
        self.faces = []
        self.current_level = 0
        self.side_len = 0
        self._create_level0()

    @abstractmethod
    def _create_level0(self):
        # this is implemented by each subclass since they have different edges, vertices and faces
        pass

    def plot_graph(self):
        """
        Plot the networkx graph of self.G.
        """
        node_labels = {i: tuple(np.round(i, 3)) for i in self.G.nodes}
        nx.draw(self.G, pos=nx.circular_layout(self.G), with_labels=True, labels=node_labels)

    def plot_points(self, ax, select_faces: set = None, projection: bool = False):
        """
        Plot the points of the icosahedron + possible division points. Colored by level at which the point was added.
        Possible to select only one or a few faces on which points are to be plotted for clarity.

        Args:
            ax: axis
            select_faces: a set of face numbers from 0 to (incl) 19, e.g. {0, 5}. If None, all faces are shown.
            projection: True if you want to plot the projected points, not the ones on surfaces of polytope
        """
        select_faces = set(range(20)) if select_faces is None else select_faces
        level_color = ["black", "red", "blue", "green"]
        for point in self.G.nodes(data=True):
            # select only points that belong to at least one of the chosen select_faces
            if len(set(point[1]["face"]).intersection(select_faces)) > 0:
                # color selected based on the level of the node
                level_node = point[1]["level"]
                if projection:
                    proj_node = point[1]["projection"]
                    ax.scatter(*proj_node[0], color=level_color[level_node], s=30)
                else:
                    ax.scatter(*point[0], color=level_color[level_node], s=30)

    def plot_edges(self, ax, select_faces=None, **kwargs):
        """
        Plot the edges between the points. Can select to display only some faces for clarity.

        Args:
            ax: axis
            select_faces: a set of face numbers from 0 to (incl) 19, e.g. {0, 5}. If None, all faces are shown.
            **kwargs: other plotting arguments
        """
        select_faces = set(range(20)) if select_faces is None else select_faces
        for edge in self.G.edges:
            faces_edge_1 = set(self.G.nodes[edge[0]]["face"])
            faces_edge_2 = set(self.G.nodes[edge[1]]["face"])
            # both the start and the end point of the edge must belong to one of the selected faces
            if len(faces_edge_1.intersection(select_faces)) > 0 and len(faces_edge_2.intersection(select_faces)) > 0:
                ax.plot(*np.array(edge).T, color="black",  **kwargs)

    def divide_edges(self):
        """
        Subdivide once. If previous faces are triangles, adds three points per face (half sides). If they are
        squares, adds 4 points per face (half sides + middle). New points will have a higher level and will
        be appropriately added to one or more faces (if on edges).
        """
        # need to keep track of what to add/remove, since we can't change the network while inside the loop
        # consists of (new_point, previous_point_1, previous_point_2) to keep all info
        nodes_to_add = []
        for node_vector in self.G.nodes():
            for neighbour_vector in self.G.neighbors(node_vector):
                # this is just to avoid doing everything twice - do it for edge A-B but not B-A
                if node_vector < neighbour_vector:
                    # new point is just the average of the previous two
                    new_point = tuple((np.array(node_vector)+np.array(neighbour_vector))/2.0)
                    nodes_to_add.append((new_point, node_vector, neighbour_vector))
        # add new nodes, add edges from cont. points to the node and delete the edge between the cont. points
        for el in nodes_to_add:
            new_point, node_vector, neighbour_vector = el
            # new point will have the set of faces that both node and neighbour vector have
            faces_node_vector = self.G.nodes[node_vector]["face"]
            faces_neighbour_vector = self.G.nodes[neighbour_vector]["face"]
            # diagonals get added twice, so this is necessary
            if new_point not in self.G.nodes:
                self.G.add_node(new_point, level=self.current_level + 1,
                                face=set(faces_node_vector).intersection(faces_neighbour_vector),
                                projection=project_grid_on_sphere(np.array(new_point)[np.newaxis, :]))
            self.G.add_edge(new_point, neighbour_vector,
                            length=unit_dist_on_sphere(self.G.nodes[new_point]["projection"],
                                                       self.G.nodes[neighbour_vector]["projection"]))
            self.G.add_edge(new_point, node_vector,
                            length=unit_dist_on_sphere(self.G.nodes[new_point]["projection"],
                                                       self.G.nodes[neighbour_vector]["projection"]))
            # self.G.remove_edge(node_vector, neighbour_vector)
        # also add edges between new nodes at distance side_len or sqrt(2)*side_len
        new_level = [x for x, y in self.G.nodes(data=True) if y['level'] == self.current_level+1]
        for new_node in new_level:
            # searching only second neighbours at appropriate level
            sec_neighbours = list(second_neighbours(self.G, new_node))
            sec_neighbours = [x for x in sec_neighbours if self.G.nodes[x]["level"] == self.current_level+1]
            for other_node in sec_neighbours:
                node_dist = np.linalg.norm(np.array(new_node)-np.array(other_node))
                # check distance criterion
                if np.isclose(node_dist, self.side_len) or np.isclose(node_dist, self.side_len*np.sqrt(2)):
                    self.G.add_edge(new_node, other_node,
                                    length=unit_dist_on_sphere(self.G.nodes[new_node]["projection"],
                                                               self.G.nodes[other_node]["projection"])
                                    )
        self.current_level += 1
        self.side_len = self.side_len / 2


class IcosahedronPolytope(Polytope):
    """
    IcosahedronGrid is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D icosahedron. It is possible to subdivide the sides, in that case a new point always appears in the
    middle of each triangle side.
    """

    def _create_level0(self):
        # DO NOT change order of points - faces will be wrong!
        # each face contains numbers - indices of vertices that border on this face
        self.faces = [[0, 11, 5],
                      [0, 5, 1],
                      [0, 1, 7],
                      [0, 7, 10],
                      [0, 10, 11],
                      [1, 5, 9],
                      [5, 11, 4],
                      [11, 10, 2],
                      [10, 7, 6],
                      [7, 1, 8],
                      [3, 9, 4],
                      [3, 4, 2],
                      [3, 2, 6],
                      [3, 6, 8],
                      [3, 8, 9],
                      [4, 9, 5],
                      [2, 4, 11],
                      [6, 2, 10],
                      [8, 6, 7],
                      [9, 8, 1]]
        side_len = 1 / np.sin(2 * pi / 5)
        self.side_len = side_len
        # create vertices
        vertices = [(-1, golden, 0), (1, golden, 0), (-1, -golden, 0), (1, -golden, 0),
                    (0, -1, golden), (0, 1, golden), (0, -1, -golden), (0, 1, -golden),
                    (golden, 0, -1), (golden, 0, 1), (-golden, 0, -1), (-golden, 0, 1)]
        vertices = np.array(vertices) * side_len / 2
        # create edges
        point_connections = _calc_edges(vertices, side_len)
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            set_of_faces = tuple(faces_i for faces_i, face in enumerate(self.faces) if i in face)
            self.G.add_node(tuple(vert), level=self.current_level, face=set_of_faces,
                            projection=project_grid_on_sphere(vert[np.newaxis, :]))
        for key, value in point_connections.items():
            for vi in value:
                self.G.add_edge(key, tuple(vi),
                                length=unit_dist_on_sphere(self.G.nodes[key]["projection"],
                                                           self.G.nodes[tuple(vi)]["projection"]))
        # just to check ...
        assert self.G.number_of_nodes() == 12
        assert self.G.number_of_edges() == 30
        for node in self.G.nodes(data=True):
            assert len(node[1]["face"]) == 5 and node[1]["level"] == 0
        self.side_len = side_len / 2


class CubePolytope(Polytope):
    """
    CubeGrid is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D cube. It is possible to subdivide the sides, in that case a new point always appears in the
    middle of a square and half of previous sides.
    """

    def _create_level0(self):
        # DO NOT change order of points - faces will be wrong!
        # each face contains numbers - indices of vertices that border on this face
        self.faces = [[0, 1, 2, 4],
                      [0, 2, 3, 6],
                      [0, 1, 3, 5],
                      [3, 5, 6, 7],
                      [1, 4, 5, 7],
                      [2, 4, 6, 7]]
        self.side_len = 2 * np.sqrt(1/3)
        # create vertices
        vertices = [(-self.side_len/2, -self.side_len/2, -self.side_len/2),
                    (-self.side_len/2, -self.side_len/2, self.side_len/2),
                    (-self.side_len/2, self.side_len/2, -self.side_len/2),
                    (self.side_len/2, -self.side_len/2, -self.side_len/2),
                    (-self.side_len/2, self.side_len/2, self.side_len/2),
                    (self.side_len/2, -self.side_len/2, self.side_len/2),
                    (self.side_len/2, self.side_len/2, -self.side_len/2),
                    (self.side_len/2, self.side_len/2, self.side_len/2)]
        vertices = np.array(vertices)
        # create edges
        point_connections = _calc_edges(vertices, self.side_len)
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            set_of_faces = tuple(faces_i for faces_i, face in enumerate(self.faces) if i in face)
            self.G.add_node(tuple(vert), level=self.current_level, face=set_of_faces,
                            projection=project_grid_on_sphere(vert[np.newaxis, :]))
        for key, value in point_connections.items():
            for vi in value:
                self.G.add_edge(key, tuple(vi),
                                length=unit_dist_on_sphere(self.G.nodes[key]["projection"],
                                                           self.G.nodes[tuple(vi)]["projection"]))
        # just to check ...
        assert self.G.number_of_nodes() == 8
        assert self.G.number_of_edges() == 24
        for node in self.G.nodes(data=True):
            assert len(node[1]["face"]) == 3 and node[1]["level"] == 0
        self.side_len = self.side_len / 2


def _calc_edges(vertices: np.ndarray, side_len: float) -> dict:
    """
    Needed for the determination of edges while setting up a polytope. If a bit weird, keep in mind this is a legacy
    function. Do not use on its own.

    Args:
        vertices: an array of polytope vertices
        side_len: everything < sqrt(2)*side_len apart will be considered a neighbour

    Returns:
        a dictionary, key is the vertex, values are its neighbours
    """
    dist = distance_matrix(vertices, vertices)
    # filter points more than 0 and less than sqrt(2)*side away from each other
    # this gives exactly triangles and squares as neighbours
    where_result = np.where((dist <= np.sqrt(2)*side_len) & (0 < dist))
    indices_min_dist = zip(where_result[0], where_result[1])
    # key is the vertex, values are its neighbours
    tree_connections = {tuple(vert): [] for vert in vertices}
    for i1, i2 in indices_min_dist:
        tree_connections[tuple(vertices[i1])].append(vertices[i2])
    return tree_connections


def second_neighbours(graph: nx.Graph, node):
    """Yield second neighbors of node in graph.
    Neighbors are not not unique!
    """
    for neighbor_list in [graph.neighbors(n) for n in graph.neighbors(node)]:
        for n in neighbor_list:
            yield n


class Grid(ABC):

    def __init__(self, N: int, *, ordered: bool = True, use_saved: bool = False, gen_alg: str = None,
                 time_generation: bool = False):
        """
        Generate a grid with one of generation algorithms.

        Args:
            gen_alg: MUST BE SET IN SUBCLASSES, algorithm name, see names given in SIX_METHOD_NAMES
            N: number of grid points
            ordered: if True order and truncate, else only truncate to N points
            use_saved: if True use saved grids if available
            time_generation: if True write out a message about time needed for generation
        """
        self.rn_gen = np.random.default_rng(DEFAULT_SEED)
        np.random.seed(DEFAULT_SEED)
        if gen_alg not in SIX_METHOD_NAMES:
            raise ValueError(f"{gen_alg} is not a valid generation algorithm name. Try 'ico', 'cube3D' ...")
        self.ordered = ordered
        self.N = N
        name_properties = {"grid_type": gen_alg, "num_grid_points": N, "ordering": ordered}
        self.standard_name = NameParser(name_properties).get_standard_name()
        self.decorator_label = f"rotation grid {self.standard_name}"
        self.grid = None
        self.time = 0
        self.nn_dist_arch = None
        self.nn_dist_cup = None
        self.short_statistics_path = f"{PATH_OUTPUT_STAT}{self.standard_name}_short_stat.txt"
        self.statistics_path = f"{PATH_OUTPUT_STAT}{self.standard_name}_full_stat.csv"
        gen_func = self.generate_and_time if time_generation else self.generate_grid
        # if this option enabled, search first if this grid has already been saved
        if use_saved:
            try:
                self.grid = np.load(f"{PATH_OUTPUT_ROTGRIDS}{self.standard_name}.npy")
            except FileNotFoundError:
                gen_func()
                self.save_grid()
        else:
            gen_func()
        assert isinstance(self.grid, np.ndarray), "A grid must be a numpy array!"
        assert self.grid.shape == (N, 3), f"Grid not of correct shape! {self.grid.shape} instead of {(N, 3)}"
        assert np.allclose(np.linalg.norm(self.grid, axis=1), 1, atol=10**(-UNIQUE_TOL))

    def get_grid(self) -> np.ndarray:
        return self.grid

    def reduce_N(self, reduce_by=1):
        if reduce_by <= 0:
            return
        self.grid = self.grid[:-reduce_by]
        self.N = self.N - reduce_by
        self.time = 0
        self.nn_dist_arch = None
        self.nn_dist_cup = None

    @abstractmethod
    def generate_grid(self):
        # order or truncate
        if self.ordered:
            self._order()
        else:
            self.grid = self.grid[:self.N]

    @time_method
    def generate_and_time(self):
        self.generate_grid()

    def _order(self):
        self.grid = order_grid_points(self.grid, self.N)

    def as_quaternion(self) -> np.ndarray:
        quaternion_seq = grid2quaternion(self.grid)
        assert isinstance(quaternion_seq, np.ndarray), "A quaternion sequence must be a numpy array!"
        assert quaternion_seq.shape == (self.N, 4), f"Quaternion sequence not of correct shape!\
                                {quaternion_seq.shape} instead of {(self.N, 4)}"
        return quaternion_seq

    def as_euler(self) -> np.ndarray:
        euler_seq = grid2euler(self.grid)
        assert isinstance(euler_seq, np.ndarray), "An Euler sequence must be a numpy array!"
        assert euler_seq.shape == (self.N, 3), f"An Euler sequence not of correct shape!\
                                {euler_seq.shape} instead of {(self.N, 33)}"
        return euler_seq

    def save_grid(self):
        np.save(f"{PATH_OUTPUT_ROTGRIDS}{self.standard_name}.{ENDING_GRID_FILES}", self.grid)

    def save_grid_txt(self):
        np.savetxt(f"{PATH_OUTPUT_ROTGRIDS}{self.standard_name}.txt", self.grid)

    def save_statistics(self, num_random: int = 100, print_message=False):
        # first message (what measure you are using)
        newline = "\n"
        m1 = f"STATISTICS: Testing the coverage of grid {self.standard_name} using {num_random} " \
             f"random points on a sphere."
        m2 = f"We select {num_random} random axes and count the number of grid points that fall within the angle" \
             f"alpha (selected from [pi / 6, 2 * pi / 6, 3 * pi / 6, 4 * pi / 6, 5 * pi / 6]) of this axis. For an" \
             f"ideally uniform grid, we expect the ratio of num_within_alpha/total_num_points to equal the ratio" \
             f"area_of_alpha_spherical_cap/area_of_sphere, which we call ideal coverage."
        stat_data, full_data = self._generate_statistics(num_rand_points=num_random)
        if print_message:
            print(m1)
            print(stat_data)
        # dealing with the file
        with open(self.short_statistics_path, "w") as f:
            f.writelines([m1, newline, newline, m2, newline, newline])
        stat_data.to_csv(self.short_statistics_path, mode="a")
        full_data.to_csv(self.statistics_path, mode="w")

    def _generate_statistics(self, num_rand_points: int = 100) -> tuple:
        # write out short version ("N points", "min", "max", "average", "SD")
        columns = ["alphas", "ideal coverages", "min coverage", "avg coverage", "max coverage", "standard deviation"]
        ratios_columns = ["coverages", "alphas", "ideal coverage"]
        alphas = [pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6]
        ratios = [[], [], []]
        sphere_surface = 4 * pi
        data = np.zeros((len(alphas), 6))  # 5 data columns for: alpha, ideal coverage, min, max, average, sd
        for i, alpha in enumerate(alphas):
            cone_area = 2 * pi * (1-np.cos(alpha))
            ideal_coverage = cone_area / sphere_surface
            actual_coverages = random_axes_count_points(self, alpha, num_random_points=num_rand_points)
            ratios[0].extend(actual_coverages)
            ratios[1].extend([alpha]*num_rand_points)
            ratios[2].extend([ideal_coverage]*num_rand_points)
            data[i][0] = alpha
            data[i][1] = ideal_coverage
            data[i][2] = np.min(actual_coverages)
            data[i][3] = np.average(actual_coverages)
            data[i][4] = np.max(actual_coverages)
            data[i][5] = np.std(actual_coverages)
        alpha_df = pd.DataFrame(data=data, columns=columns)
        alpha_df = alpha_df.set_index("alphas")
        ratios_df = pd.DataFrame(data=np.array(ratios).T, columns=ratios_columns)
        return alpha_df, ratios_df


def order_grid_points(grid: np.ndarray, N: int, start_i: int = 1) -> np.ndarray:
    """
    You are provided with a (possibly) unordered grid and return a grid with N points ordered in such a way that
    these N points have the best possible coverage.

    Args:
        grid: grid, array of shape (L, 3) to be ordered where L should be >= N
        N: number of grid points wished at the end
        start_i: from which index to start ordering (in case the first i elements already ordered)

    Returns:
        an array of shape (N, 3) ordered in such a way that these N points have the best possible coverage.
    """
    if N > len(grid):
        print(f"Warning! N>len(grid)! Only {len(grid)} points will be returned!")
    for index in range(start_i, min(len(grid), N)):
        grid = select_next_gridpoint(grid, index)
    return grid[:N]


def project_grid_on_sphere(grid: np.ndarray) -> np.ndarray:
    """
    A grid can be seen as a collection of vectors to gridpoints. If a vector is scaled to 1, it will represent a point
    on a unit sphere in d-1 dimensions. This function normalizes the vectors, creating vectors pointing to the
    surface of a d-1 dimensional sphere.

    Args:
        grid: a (N, d) array where each row represents the coordinates of a grid point

    Returns:
        a (N, d) array where each row has been scaled to length 1
    """
    largest_abs = np.max(np.abs(grid), axis=1)[:, np.newaxis]
    grid = np.divide(grid, largest_abs)
    norms = np.linalg.norm(grid, axis=1)[:, np.newaxis]
    return np.divide(grid, norms)


def select_next_gridpoint(set_grid_points, i):
    """
    Provide a set of grid points where the first i are already sorted. Find the best next gridpoint out of points
    in set_grid_points[i:]

    Args:
        set_grid_points: grid, array of shape (L, 3) where elements up to i are already ordered
        i: index how far the array is already ordered (up to bun not including i).

    Returns:
        set_grid_points where the ith element in swapped with the best possible next grid point
    """
    distances = cdist(set_grid_points[i:], set_grid_points[:i], metric="cosine")
    distances.sort()
    nn_dist = distances[:, 0]
    index_max = np.argmax(nn_dist)
    set_grid_points[[i, i + index_max]] = set_grid_points[[i + index_max, i]]
    return set_grid_points


class RandomQGrid(Grid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, gen_alg="randomQ", **kwargs)

    def generate_grid(self):
        result = np.zeros((self.N, 4))
        random_num = self.rn_gen.random((self.N, 3))
        result[:, 0] = np.sqrt(1 - random_num[:, 0]) * np.sin(2 * pi * random_num[:, 1])
        result[:, 1] = np.sqrt(1 - random_num[:, 0]) * np.cos(2 * pi * random_num[:, 1])
        result[:, 2] = np.sqrt(random_num[:, 0]) * np.sin(2 * pi * random_num[:, 2])
        result[:, 3] = np.sqrt(random_num[:, 0]) * np.cos(2 * pi * random_num[:, 2])
        self.grid = quaternion2grid(result)
        # No super call because ordering not needed for random points and the number of points is exact!


class RandomEGrid(Grid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, gen_alg="randomE", **kwargs)

    def generate_grid(self):
        euler_angles = 2 * pi * self.rn_gen.random((self.N, 3))
        self.grid = euler2grid(euler_angles)
        # No super call because ordering not needed for random points and the number of points is exact!


class SystemEGrid(Grid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, gen_alg="systemE", **kwargs)

    def generate_grid(self):
        self.grid = []
        num_points = 1
        while len(self.grid) < self.N:
            phis = np.linspace(0, 2 * pi, num_points)
            thetas = np.linspace(0, 2 * pi, num_points)
            psis = np.linspace(0, 2 * pi, num_points)
            euler_meshgrid = np.array(np.meshgrid(*(phis, thetas, psis)), dtype=float)
            euler_meshgrid = euler_meshgrid.reshape((3, -1)).T
            # convert to a grid
            self.grid = euler2grid(euler_meshgrid)
            self.grid = np.unique(np.round(self.grid, UNIQUE_TOL), axis=0)
            num_points += 1
        super().generate_grid()


class Cube4DGrid(Grid):

    def __init__(self, N: int, **kwargs):
        self.d = 4  # dimensions
        super().__init__(N, gen_alg="cube4D", **kwargs)

    def generate_grid(self):
        self.grid = []
        num_divisions = 1
        state_before = np.random.get_state()
        while len(self.grid) < self.N:
            grid_qua = self._full_d_dim_grid()
            grid_qua = self._select_only_faces(grid_qua)
            grid_qua = project_grid_on_sphere(grid_qua)
            # select only half the sphere
            grid_qua = grid_qua[grid_qua[:, self.d - 1] >= 0, :]
            # convert to grid
            self.grid = quaternion2grid(grid_qua)
            self.grid = np.unique(np.round(self.grid, UNIQUE_TOL), axis=0)
            num_divisions += 1
        np.random.set_state(state_before)
        super().generate_grid()

    def _full_d_dim_grid(self, cheb: bool = False, change_start: float = 0, change_end: float = 0,
                         dtype=np.float64) -> np.ndarray:
        """
        This is a function to create a classical grid of a d-dimensional cube. It creates a grid over the entire
        (hyper)volume of the (hyper)cube.

        This is a unit cube between -sqrt(1/d) and sqrt(1/d) in all dimensions where d = num of dimensions.

        Args:
            cheb: use Chebyscheff points instead of equally spaced points
            change_start: add or subtract from -sqrt(1/d) as the start of the grid
            change_end: add or subtract from sqrt(1/d) as the end of the grid
            dtype: forwarded to linspace while creating a grid

        Returns:
            a meshgrid of dimension (d, n, n, .... n) where n is repeated d times
        """
        if cheb:
            from numpy.polynomial.chebyshev import chebpts1
            side = chebpts1(self.N)
        else:
            # np.sqrt(1/d)
            side = np.linspace(-1 + change_start, 1 - change_end, self.N, dtype=dtype)
        # repeat the same n points d times and then make a new line of the array every d elements
        sides = np.tile(side, self.d)
        sides = sides[np.newaxis, :].reshape((self.d, self.N))
        # create a grid by meshing every line of the sides array
        return np.array(np.meshgrid(*sides))

    def _select_only_faces(self, grid: np.ndarray):
        """
        Take a meshgrid (d, n, n, ... n)  and return an array of points (N, d) including only the points that
        lie on the faces of the grid, so the edge points in at least one of dimensions.

        Args:
            grid: numpy array (d, n, n, ... n) containing grid points

        Returns:
            points (N, d) where N is the number of edge points and d the dimension
        """
        assert self.d == len(grid)
        set_grids = []
        for swap_i in range(self.d):
            meshgrid_swapped = np.swapaxes(grid, axis1=1, axis2=(1 + swap_i))
            set_grids.append(meshgrid_swapped[:, 0, ...])
            set_grids.append(meshgrid_swapped[:, -1, ...])

        result = np.hstack(set_grids).reshape((self.d, -1)).T
        return np.unique(result, axis=0)


class Polyhedron3DGrid(Grid):

    def __init__(self, N: int, polyhedron, **kwargs):
        self.polyhedron = polyhedron()
        super().__init__(N, **kwargs)

    def generate_grid(self):
        while self.polyhedron.G.number_of_nodes() < self.N:
            self.polyhedron.divide_edges()
        self.grid = np.array([y["projection"] for x, y in self.polyhedron.G.nodes(data=True)]).squeeze()
        np.random.shuffle(self.grid)
        super().generate_grid()


class Cube3DGrid(Polyhedron3DGrid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, polyhedron=CubePolytope, gen_alg="cube3D", **kwargs)


class IcoGrid(Polyhedron3DGrid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, polyhedron=IcosahedronPolytope, gen_alg="ico", **kwargs)


def build_grid(grid_type: str, N: int, **kwargs) -> Grid:
    name2grid = {"randomQ": RandomQGrid,
                 "randomE": RandomEGrid,
                 "cube4D": Cube4DGrid,
                 "systemE": SystemEGrid,
                 "cube3D": Cube3DGrid,
                 "ico": IcoGrid}
    if grid_type not in name2grid.keys():
        raise ValueError(f"{grid_type} is not a valid grid type. Try 'ico', 'cube3D' ...")
    grid_obj = name2grid[grid_type]
    return grid_obj(N, **kwargs)