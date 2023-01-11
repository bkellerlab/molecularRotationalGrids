import os.path
from abc import ABC, abstractmethod
from itertools import product

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import pandas as pd

from molgri.analysis import random_quaternions, random_quaternions_count_points
from molgri.constants import UNIQUE_TOL, EXTENSION_GRID_FILES
from molgri.grids import Grid, project_grid_on_sphere, build_grid
from molgri.parsers import GridNameParser
from molgri.paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_STAT
from molgri.rotations import rotation2grid, grid2rotation, grid2quaternion, grid2euler
from molgri.utils import dist_on_sphere


class RotationsObject(ABC):

    def __init__(self, N: int = None, gen_algorithm: str = None, use_saved=True):
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.rotations = None
        self.N = N
        self.gen_algorithm = gen_algorithm
        self.standard_name = f"{gen_algorithm}_{N}"
        self.short_statistics_path = f"{PATH_OUTPUT_STAT}{self.standard_name}_short_stat_rotobj.txt"
        self.statistics_path = f"{PATH_OUTPUT_STAT}{self.standard_name}_full_stat_rotobj.csv"
        if use_saved:
            try:
                grid_x = np.load(f"{PATH_OUTPUT_ROTGRIDS}{gen_algorithm}_{N}_x.{EXTENSION_GRID_FILES}")
                grid_y = np.load(f"{PATH_OUTPUT_ROTGRIDS}{gen_algorithm}_{N}_y.{EXTENSION_GRID_FILES}")
                grid_z = np.load(f"{PATH_OUTPUT_ROTGRIDS}{gen_algorithm}_{N}_z.{EXTENSION_GRID_FILES}")
                self.from_grids(grid_x, grid_y, grid_z)
            except FileNotFoundError:
                self.gen_rotations(self.N, self.gen_algorithm)
                self.save_all()
        else:
            self.gen_rotations(self.N, self.gen_algorithm)
            self.save_all()

    @abstractmethod
    def gen_rotations(self, N: int = None, gen_algorithm: str = None):
        pass

    def _order_rotations(self):
        """
        You are provided with a (possibly) unordered array of rotations saved in self.rotations. You must re-order
        it so that the coverage is maximised at every step.

        Additionally, truncate at self.N.
        """
        rot_quaternions = self.rotations.as_quat()
        if self.N > len(rot_quaternions):
            raise ValueError(f"N>len(grid)! Only {len(rot_quaternions)} points can be returned!")
        for index in range(1, self.N):
            rot_quaternions = select_next_rotation(rot_quaternions, index)
        rot_quaternions = rot_quaternions[:self.N]
        self.rotations = Rotation.from_quat(rot_quaternions)

    def _select_unique_rotations(self):
        rot_matrices = self.rotations.as_matrix()
        rot_matrices = np.unique(np.round(rot_matrices, UNIQUE_TOL), axis=0)
        self.rotations = Rotation.from_matrix(rot_matrices)

    def from_rotations(self, rotations: Rotation):
        self.rotations = rotations
        self.grid_x, self.grid_y, self.grid_z = rotation2grid(rotations)
        self._determine_N()

    def from_grids(self, grid_x: NDArray, grid_y: NDArray, grid_z: NDArray):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.rotations = self.as_rotation_object()
        self._determine_N()

    def _determine_N(self):
        if self.grid_x is None and self.grid_y is None and self.grid_z is None:
            self.N = None
        else:
            assert len(self.grid_x) == len(self.grid_y) == len(self.grid_z)
            self.N = len(self.grid_x)

    def as_rotation_object(self) -> Rotation:
        return grid2rotation(self.grid_x, self.grid_y, self.grid_z)

    def as_quaternion(self) -> NDArray:
        return grid2quaternion(self.grid_x, self.grid_y, self.grid_z)

    def as_euler(self) -> NDArray:
        return grid2euler(self.grid_x, self.grid_y, self.grid_z)

    def save_all(self):
        subgrids = (self.grid_x, self.grid_y, self.grid_z)
        labels = ("x", "y", "z")
        for label, sub_grid in zip(labels, subgrids):
            file_name = f"{self.standard_name}_{label}"
            np.save(f"{PATH_OUTPUT_ROTGRIDS}{file_name}.{EXTENSION_GRID_FILES}", sub_grid)

    def save_statistics(self, num_random: int = 100, print_message=False, alphas=None):
        if alphas is None:
            alphas = [pi / 6, 2 * pi / 6, 3 * pi / 6, 4 * pi / 6, 5 * pi / 6]
        # first message (what measure you are using)
        newline = "\n"
        m1 = f"STATISTICS: Testing the coverage of grid {self.standard_name} using {num_random} " \
             f"random points on a sphere."
        m2 = f"We select {num_random} random axes and count the number of grid points that fall within the angle" \
             f"alpha (selected from [pi / 6, 2 * pi / 6, 3 * pi / 6, 4 * pi / 6, 5 * pi / 6]) of this axis. For an" \
             f"ideally uniform grid, we expect the ratio of num_within_alpha/total_num_points to equal the ratio" \
             f"area_of_alpha_spherical_cap/area_of_sphere, which we call ideal coverage."
        stat_data, full_data = self._generate_statistics(alphas, num_rand_points=num_random)
        if print_message:
            print(m1)
            print(stat_data)
        # dealing with the file
        with open(self.short_statistics_path, "w") as f:
            f.writelines([m1, newline, newline, m2, newline, newline])
        if os.path.exists(self.statistics_path):
            os.remove(self.statistics_path)
        stat_data.to_csv(self.short_statistics_path, mode="a")
        full_data.to_csv(self.statistics_path, mode="w")

    def _generate_statistics(self, alphas, num_rand_points: int = 1000) -> tuple:
        # write out short version ("N points", "min", "max", "average", "SD"
        columns = ["alphas", "ideal coverages", "min coverage", "avg coverage", "max coverage", "standard deviation"]
        ratios_columns = ["coverages", "alphas", "ideal coverage"]
        ratios = [[], [], []]
        sphere_surface = 2*pi**2 # full 4D sphere has area 2pi^2 r^3
        data = np.zeros((len(alphas), 6))  # 5 data columns for: alpha, ideal coverage, min, max, average, sd
        for i, alpha in enumerate(alphas):
            # explanation: see https://scialert.net/fulltext/?doi=ajms.2011.66.70&org=11
            cone_area = 1/2 * sphere_surface * (2*alpha - np.sin(2*alpha))/np.pi # cone area of 4D cone is
            ideal_coverage = cone_area / sphere_surface
            actual_coverages = random_quaternions_count_points(self.rotations.as_quat(), alpha,
                                                               num_random_points=num_rand_points)
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


def select_next_rotation(quaternion_list, i):
    """
    Provide an array of quaternions where the first i are already sorted. Find the best next quaternion out of points
    in quaternion_list[i:] to maximise coverage

    Args:
        quaternion_list: array of shape (L, 4) where elements up to i are already ordered
        i: index how far the array is already ordered (up to bun not including i).

    Returns:
        set_grid_points where the ith element in swapped with the best possible next grid point
    """
    distances = cdist(quaternion_list[i:], quaternion_list[:i], metric="cosine") # TODO: is this right?
    distances.sort()
    nn_dist = distances[:, 0]
    index_max = np.argmax(nn_dist)
    quaternion_list[[i, i + index_max]] = quaternion_list[[i + index_max, i]]
    return quaternion_list


class RandomQRotations(RotationsObject):

    def gen_rotations(self, N: int = None, gen_algorithm="randomQ"):
        assert N is not None, "Select the number of points N!"
        quaternions = random_quaternions(N)
        self.from_rotations(Rotation.from_quat(quaternions))


class SystemERotations(RotationsObject):

    def gen_rotations(self, N: int = None, gen_algorithm="randomQ"):
        assert N is not None, "Select the number of points N!"
        num_points = 1
        rot_matrices = []
        while len(rot_matrices) < N:
            phis = np.linspace(0, 2 * pi, num_points)
            thetas = np.linspace(0, 2 * pi, num_points)
            psis = np.linspace(0, 2 * pi, num_points)
            euler_meshgrid = np.array(np.meshgrid(*(phis, thetas, psis)), dtype=float)
            euler_meshgrid = euler_meshgrid.reshape((3, -1)).T
            self.rotations = Rotation.from_euler("ZYX", euler_meshgrid)
            # remove non-unique rotational matrices
            self._select_unique_rotations()
            rot_matrices = self.rotations.as_matrix()
            num_points += 1
        self._order_rotations()
        # convert to a grid
        self.from_rotations(self.rotations)
        # self.grid = euler2grid(euler_meshgrid)


class RandomERotations(RotationsObject):

    def gen_rotations(self, N: int = None, gen_algorithm="randomE"):
        euler_angles = 2 * pi * np.random.random((N, 3))
        self.from_rotations(Rotation.from_euler("ZYX", euler_angles))


class Cube4DRotations(RotationsObject):

    def __init__(self, *args, **kwargs):
        self.d = 4
        super().__init__(*args, **kwargs)

    def gen_rotations(self, N: int = None, gen_algorithm = "cube4D"):
        self.N = N
        rot_matrices = []
        num_divisions = 1
        state_before = np.random.get_state()
        while len(rot_matrices) < self.N:
            grid_qua = self._full_d_dim_grid()
            grid_qua = self._select_only_faces(grid_qua)
            grid_qua = project_grid_on_sphere(grid_qua)
            # select only half the sphere
            grid_qua = grid_qua[grid_qua[:, self.d - 1] >= 0, :]
            self.rotations = Rotation.from_quat(grid_qua)
            # remove non-unique rotational matrices
            self._select_unique_rotations()
            rot_matrices = self.rotations.as_matrix()
            num_divisions += 1
        np.random.set_state(state_before)
        self._order_rotations()
        self.from_rotations(self.rotations)

    def _full_d_dim_grid(self, dtype=np.float64) -> np.ndarray:
        """
        This is a function to create a classical grid of a d-dimensional cube. It creates a grid over the entire
        (hyper)volume of the (hyper)cube.

        This is a unit cube between -sqrt(1/d) and sqrt(1/d) in all dimensions where d = num of dimensions.

        Args:
            dtype: forwarded to linspace while creating a grid

        Returns:
            a meshgrid of dimension (d, n, n, .... n) where n is repeated d times
        """
        # TODO: could you make this less memory-intensive?? not create meshgrid?? only meshgrid in d-1 dim?
        side = np.linspace(-1, 1, self.N, dtype=dtype)
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


class ZeroRotations(RotationsObject):

    def gen_rotations(self, N=1, gen_algorithm="zero"):
        self.N = 1
        rot_matrix = np.eye(3)
        rot_matrix = rot_matrix[np.newaxis, :]
        self.rotations = Rotation.from_matrix(rot_matrix)
        self.from_rotations(self.rotations)


# this is only a test, not sure if right
class IcoRotations(RotationsObject):

    def gen_rotations(self, N = None, gen_algorithm="ico"):
        ico_grid = build_grid(N, gen_algorithm).get_grid()
        self.from_grids(ico_grid, ico_grid, ico_grid)


class Cube3DRotations(RotationsObject):

    def gen_rotations(self, N = None, gen_algorithm="cube3D"):
        cube3d_grid = build_grid(N, gen_algorithm).get_grid()
        self.from_grids(cube3d_grid, cube3d_grid, cube3d_grid)



class Polytope(ABC):

    def __init__(self, d: int = 3):
        """
        A polytope is a 3-dim object consisting of a set of vertices and connections between them (edges) saved
        in self.G property.
        """
        self.G = nx.Graph()
        self.faces = []
        self.current_level = 0
        self.side_len = 0
        self.d = d
        self._create_level0()

    @abstractmethod
    def _create_level0(self):
        """This is implemented by each subclass since they have different edges, vertices and faces"""

    def plot_graph(self):
        """
        Plot the networkx graph of self.G.
        """
        node_labels = {i: tuple(np.round(i, 3)) for i in self.G.nodes}
        nx.draw_networkx(self.G, pos=nx.circular_layout(self.G), with_labels=True, labels=node_labels)

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
                                face=None,
                                projection=project_grid_on_sphere(np.array(new_point)[np.newaxis, :]))
            self.G.add_edge(new_point, neighbour_vector,
                            length=dist_on_sphere(self.G.nodes[new_point]["projection"],
                                                  self.G.nodes[neighbour_vector]["projection"]))
            self.G.add_edge(new_point, node_vector,
                            length=dist_on_sphere(self.G.nodes[new_point]["projection"],
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
                                    length=dist_on_sphere(self.G.nodes[new_node]["projection"],
                                                          self.G.nodes[other_node]["projection"])
                                    )
        self.current_level += 1
        self.side_len = self.side_len / 2


class Cube4DPolytope(Polytope):
    """
    CubeGrid is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D cube. It is possible to subdivide the sides, in that case a new point always appears in the
    middle of a square and half of previous sides.
    """

    def __init__(self):
        super().__init__(d=4)

    def _create_level0(self):
        # DO NOT change order of points - faces will be wrong!
        # each face contains numbers - indices of vertices that border on this face
        self.faces = None
        self.side_len = 2 * np.sqrt(1/self.d)
        # create vertices
        vertices = list(product((-self.side_len/2, self.side_len/2), repeat=4))
        assert len(vertices) == 16
        assert np.all([np.isclose(x, 0.5) or np.isclose(x, -0.5) for row in vertices for x in row ])
        assert len(set(vertices)) == 16
        vertices = np.array(vertices)
        # for i, vert1 in enumerate(vertices):
        #     for j, vert2 in enumerate(vertices[i+1:]):
        #         print(np.sum(vert1 - vert2))
        # create edges
        point_connections = _calc_edges(vertices, self.side_len)
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            #set_of_faces = tuple(faces_i for faces_i, face in enumerate(self.faces) if i in face)
            self.G.add_node(tuple(vert), level=self.current_level, face=None,
                            projection=project_grid_on_sphere(vert[np.newaxis, :]))
        for key, value in point_connections.items():
            for vi in value:
                self.G.add_edge(key, tuple(vi),
                                length=dist_on_sphere(self.G.nodes[key]["projection"],
                                                      self.G.nodes[tuple(vi)]["projection"]))
        # just to check ...
        assert self.G.number_of_nodes() == 16
        #assert self.G.number_of_edges() == 32
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
    # each node hast 4 closest points
    where_result = np.where((dist <= np.sqrt(2)*side_len) & (0 < dist))
    indices_min_dist = zip(where_result[0], where_result[1])
    # key is the vertex, values are its neighbours
    tree_connections = {tuple(vert): [] for vert in vertices}
    for i1, i2 in indices_min_dist:
        tree_connections[tuple(vertices[i1])].append(vertices[i2])
    return tree_connections


def second_neighbours(graph: nx.Graph, node):
    """Yield second neighbors of node in graph. Ignore second neighbours that are also first neighbours.
    Second neighbors may repeat!

    Example:

        5------6
        |      |
        2 ---- 1 ---- 3 ---- 7
               |      |
               \--8--/

    First neighbours of 1: 2, 6, 3, 8
    Second neighbours of 1: 5, 7
    """
    direct_neighbours = list(graph.neighbors(node))
    for neighbor_list in [graph.neighbors(n) for n in direct_neighbours]:
        for n in neighbor_list:
            if n != node and n not in direct_neighbours:
                yield n


class Polyhedron4DGrid(RotationsObject):

    def __init__(self, polyhedron, *args, **kwargs):
        self.polyhedron = polyhedron()
        super().__init__(*args, **kwargs)

    def gen_rotations(self, N: int = None, gen_algorithm: str = None):
        while self.polyhedron.G.number_of_nodes() < N:
            self.polyhedron.divide_edges()
        rotations = np.array([y["projection"] for x, y in self.polyhedron.G.nodes(data=True)]).squeeze()
        self.rotations = Rotation.from_quat(rotations)
        self._order_rotations()
        self.from_rotations(self.rotations)


class Cube4DRotations2(Polyhedron4DGrid):

    def __init__(self, N: int, gen_algorithm, **kwargs):
        super().__init__(polyhedron=Cube4DPolytope, N=N, gen_algorithm=gen_algorithm, **kwargs)

    def gen_rotations(self, N: int = None, gen_algorithm = "cube4D"):
        super().gen_rotations(N, gen_algorithm)


def build_rotations_from_name(grid_name: str, b_or_o="o", use_saved=True, **kwargs) -> RotationsObject:
    gnp = GridNameParser(grid_name, b_or_o)
    return build_rotations(gnp.N, gnp.algo, use_saved=use_saved, **kwargs)


def build_rotations(N: int, algo: str, **kwargs) -> RotationsObject:
    name2rotation = {"randomQ": RandomQRotations,
                     "systemE": SystemERotations,
                     "randomE": RandomERotations,
                     "cube4D": Cube4DRotations2,
                     "zero": ZeroRotations,
                     "ico": IcoRotations,
                     "cube3D": Cube3DRotations
                     }
    if algo not in name2rotation.keys():
        raise ValueError(f"Algorithm {algo} is not a valid grid type. "
                         f"Try 'ico', 'cube3D' ...")
    assert isinstance(N, int), f"Number of grid points must be an integer, currently N={N}"
    assert N >= 0, f"Number of grid points cannot be negative, currently N={N}"
    rot_obj = name2rotation[algo](N, gen_algorithm=algo, **kwargs)
    return rot_obj


if __name__ == "__main__":
    rot_obj = build_rotations(15, "cube4D", use_saved=False)
