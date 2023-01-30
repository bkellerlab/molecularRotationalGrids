"""
Polytopes are networkX graphs representing bodies in 3- or 4D. Specifically, we implement the 3D polytopes
 icosahedron and cube and the 4D polytope hypercube. As a start, vertices of the polytope are added as
  nodes and edges as connections between them. For the cube, a point is also added in the middle of each face diagonal
  and the edges to the four vertices are added. It's important that polytopes can also be subdivided in smaller units
 so that grids with larger numbers of points can be created. For this, use divide_edges() command. 3D polytopes
 (polyhedra) can also be plotted - points and edges separately, level of discretisation can determine the color.
"""

from abc import ABC, abstractmethod
from itertools import product, combinations

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import pi, golden
from scipy.spatial import distance_matrix

from molgri.assertions import is_array_with_d_dim_r_rows_c_columns, all_row_norms_equal_k, form_square
from molgri.space.utils import dist_on_sphere, normalise_vectors


class Polytope(ABC):

    def __init__(self, d: int = 3):
        """
        A polytope is a d-dim object consisting of a set of nodes (vertices) and connections between them (edges) saved
        in self.G (graph).

        The basic polytope will be created when the object is initiated. All further divisions should be performed
        with the function self.divide_edges()

        Args:
            d: number of dimensions
        """
        self.G = nx.Graph()
        # faces can, but not need to be saved. They consist of sub-lists that contain indices of points that belong to
        # a specific face. Useful for plotting to display only one/a few faces
        self.faces = []
        self.current_level = 0              # level of division
        self.side_len = 0
        self.d = d
        self._create_level0()

    @abstractmethod
    def _create_level0(self):
        """This is implemented by each subclass since they have different edges, vertices and faces"""

    def plot_graph(self, with_labels=True):
        """
        Plot the networkx graph of self.G.
        """
        node_labels = {i: tuple(np.round(i, 3)) for i in self.G.nodes}
        nx.draw_networkx(self.G, pos=nx.circular_layout(self.G), with_labels=with_labels, labels=node_labels)

    def get_node_coordinates(self) -> NDArray:
        """
        Get an array in which each row represents the self.d - dimensional coordinates of the node. These are the
        points on the faces of the polytope, their norm may not be one (actually, will only be one at level 0).
        """
        nodes = self.G.nodes
        nodes = np.array(nodes)
        is_array_with_d_dim_r_rows_c_columns(nodes, d=2, r=self.G.number_of_nodes(), c=self.d)
        return nodes

    def get_projection_coordinates(self) -> NDArray:
        """
        Get an array in which each row represents the self.d - dimensional coordinates of the node projected on a
        self.d-dimensional unit sphere. These points must have norm 1
        """
        projections_dict = nx.get_node_attributes(self.G, "projection")
        nodes = self.get_node_coordinates()
        projections = np.zeros((len(nodes), self.d))
        for i, n in enumerate(nodes):
            projections[i] = projections_dict[tuple(n)]
        # assertions
        is_array_with_d_dim_r_rows_c_columns(projections, d=2, r=self.G.number_of_nodes(), c=self.d)
        all_row_norms_equal_k(projections, 1)
        return projections

    @abstractmethod
    def divide_edges(self):
        """
        Subdivide once. If previous faces are triangles, adds one point at mid-point of each edge. If they are
        squares, adds one point at mid-point of each edge + 1 in the middle of the face. New points will have a higher
        level attribute.
        """
        # need to keep track of what to add/remove, since we can't change the network while inside the loop
        # consists of (new_point, previous_point_1, previous_point_2) to keep all info
        nodes_to_add = self._find_new_nodes()
        # add new nodes, add edges from cont. points to the node and delete the edge between the cont. points
        self._split_edges(nodes_to_add)

    def _start_new_level(self):
        self.current_level += 1
        self.side_len = self.side_len / 2

    def _add_square_diagonal_nodes(self):
        """
        Search for any nodes that form a square with 2 neighbour nodes and 1 second neighbour node. If you find them,
        add the middle point (average of the four) to nodes along with edges from the middle to the points that
        form the square.
        """
        # here save the new node along with the 4 "father nodes" so you can later build all connections
        new_nodes = []
        for node in self.G.nodes:
            neighbours = list(self.G.neighbors(node))
            sec_neighbours = list(second_neighbours(self.G, node))
            # prepare all combinations
            # a possibility for a square: node, two neighbours, one second neighbour
            for sn in sec_neighbours:
                n_choices = list(combinations(neighbours, 2))
                for two_n in n_choices:
                    n1, n2 = two_n
                    points = [node, sn, n1, n2]
                    array_points = np.array(points)
                    if form_square(array_points):
                        midpoint = np.average(array_points, axis=0)
                        new_nodes.append([tuple(midpoint), *points])
        # add new nodes and edges
        for new_obj in new_nodes:
            new_node = new_obj[0]
            new_neighbours = new_obj[1:]
            # just in case repetition happens
            if new_node not in self.G.nodes:
                self.G.add_node(new_node, level=self.current_level, face=self._find_face(new_neighbours),
                                projection=normalise_vectors(np.array(new_node)))
                for neig in new_neighbours:
                    length = dist_on_sphere(self.G.nodes[new_node]["projection"],
                                            self.G.nodes[neig]["projection"])
                    self.G.add_edge(new_node, neig, length=length)

    def _add_edges_of_len(self, edge_len: float, wished_levels: list = None):
        """Among points on the most current level, search for second neighbours and add an edge between them
        if their distance is close to edge_len


        conditions:
         - must be second neighbours,
         - at least one in the last level
         - must be on the same face
         - must be a specified length apart

        Args:
            edge_len: length of edge on the polyhedron surface that is condition for adding edges
        """
        if wished_levels is None:
            wished_levels = [self.current_level+1, self.current_level+1]
        else:
            wished_levels.sort(reverse=True)
        assert len(wished_levels) == 2
        new_level = [x for x, y in self.G.nodes(data=True) if y['level'] == wished_levels[0]]
        #new_edges = []  # node1, node2, dist
        for new_node in new_level:
            # searching only second neighbours at appropriate level
            sec_neighbours = list(second_neighbours(self.G, new_node))
            sec_neighbours = [x for x in sec_neighbours if self.G.nodes[x]["level"] == wished_levels[1]]
            for other_node in sec_neighbours:
                node_dist = np.linalg.norm(np.array(new_node)-np.array(other_node))
                # check face criterion
                if self._find_face([new_node, other_node]):
                    # check distance criterion
                    if np.isclose(node_dist, edge_len):
                        length = dist_on_sphere(self.G.nodes[new_node]["projection"],
                                                self.G.nodes[other_node]["projection"])
                        # just to make sure that you don't add same edge in 2 different directions
                        if new_node < other_node:
                            self.G.add_edge(new_node, other_node, length=length)
                        else:
                            self.G.add_edge(other_node, new_node, length=length)

    def _find_new_nodes(self) -> list:
        """
        Helper function to find all new nodes - in between two neighbouring nodes.

        Returns:
            a list in which every element is a tuple of three elements:
            (new_point, old_point1, old_point2)
            where new_point is the arithmetic average of both old points.
            The three points are all self.d-dimensional tuples
        """
        nodes_to_add = []
        for node_vector in self.G.nodes():
            for neighbour_vector in self.G.neighbors(node_vector):
                # this is just to avoid doing everything twice - do it for edge A-B but not B-A
                if node_vector < neighbour_vector:
                    # new point is just the average of the previous two
                    new_point = tuple((np.array(node_vector)+np.array(neighbour_vector))/2.0)
                    nodes_to_add.append((new_point, node_vector, neighbour_vector))
        return nodes_to_add

    def _find_face(self, node_list: list) -> set:
        """
        Find the face that is common between all nodes in the list.

        Args:
            node_list: a list in which each element is a tuple of coordinates that has beed added to self.G as a node

        Returns:
            the set of faces (may be empty) that all nodes in this list share
        """
        try:
            face = set(self.G.nodes[node_list[0]]["face"])
        # if faces are not defined, return empty set every time
        except TypeError:
            return set()
        for neig in node_list[1:]:
            faces_neighbour_vector = self.G.nodes[neig]["face"]
            face = face.intersection(set(faces_neighbour_vector))
        return face

    def _split_edges(self, nodes_to_add):
        """
        Use the output from self._find_new_nodes(), enter these points in the graph, and replace the connection
        between the old points with two new connections: old_point1 ---- new_point and new_point ---- old_point2
        """
        for el in nodes_to_add:
            new_point, node_vector, neighbour_vector = el
            # new point will have the set of faces that both node and neighbour vector have
            if self.d <= 3:
                face = self._find_face([node_vector, neighbour_vector])
            # for higher dimensions we don't bother with faces since plotting is difficult anyway
            else:
                face = None
            # diagonals get added twice, so this is necessary
            if new_point not in self.G.nodes:
                self.G.add_node(new_point, level=self.current_level + 1,
                                face=face,
                                projection=normalise_vectors(np.array(new_point)))
            # add the two new connections: old_point1 ---- new_point and new_point ---- old_point2
            self.G.add_edge(new_point, neighbour_vector,
                            length=dist_on_sphere(self.G.nodes[new_point]["projection"],
                                                  self.G.nodes[neighbour_vector]["projection"]))
            self.G.add_edge(new_point, node_vector,
                            length=dist_on_sphere(self.G.nodes[new_point]["projection"],
                                                  self.G.nodes[neighbour_vector]["projection"]))
            # remove the old connection: old_point1 -------- old_point2
            self.G.remove_edge(node_vector, neighbour_vector)

    def _remove_edges_of_len(self, k: float):
        """
        Remove all edges from self.G that have the length k (or close to k if float)
        """
        for edge in self.G.edges:
            n1, n2 = edge
            if np.isclose(np.linalg.norm(np.abs(np.array(n2)-np.array(n1))), k):
                self.G.remove_edge(n1, n2)


class Polyhedron(Polytope, ABC):

    def __init__(self):
        """
        Polyhedron is a polytope of exactly three dimensions. The benefit: it can be plotted.
        """
        super().__init__(d=3)

    def plot_points(self, ax: Axes3D, select_faces: set = None, projection: bool = False):
        """
        Plot the points of the polytope + possible division points. Colored by level at which the point was added.
        Possible to select only one or a few faces on which points are to be plotted for clarity.

        Args:
            ax: axis
            select_faces: a set of face numbers that can range from 0 to (incl) len(self.faces), e.g. {0, 5}.
                          If None, all faces are shown.
            projection: True if you want to plot the projected points, not the ones on surfaces of polytope
        """
        select_faces = set(range(len(self.faces))) if select_faces is None else select_faces
        level_color = ["black", "red", "blue", "green"]
        for point in self.G.nodes(data=True):
            # select only points that belong to at least one of the chosen select_faces
            if len(set(point[1]["face"]).intersection(select_faces)) > 0:
                # color selected based on the level of the node
                level_node = point[1]["level"]
                if projection:
                    proj_node = point[1]["projection"]
                    ax.scatter(*proj_node.T, color=level_color[level_node], s=30)
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
        if self.d > 3:
            raise ValueError("Points can only be plotted for polyhedra of up to 3 dimensions.")
        select_faces = set(range(20)) if select_faces is None else select_faces
        for edge in self.G.edges:
            faces_edge_1 = set(self.G.nodes[edge[0]]["face"])
            faces_edge_2 = set(self.G.nodes[edge[1]]["face"])
            # both the start and the end point of the edge must belong to one of the selected faces
            if len(faces_edge_1.intersection(select_faces)) > 0 and len(faces_edge_2.intersection(select_faces)) > 0:
                ax.plot(*np.array(edge).T, color="black",  **kwargs)


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
        point_connections = _calc_edges(vertices, self.side_len)
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            self.G.add_node(tuple(vert), level=self.current_level, face=None,
                            projection=normalise_vectors(np.array(vert)))
        for key, value in point_connections.items():
            for vi in value:
                self.G.add_edge(key, tuple(vi),
                                length=dist_on_sphere(self.G.nodes[key]["projection"],
                                                      self.G.nodes[tuple(vi)]["projection"]))
        # create diagonal nodes # TODO: maybe should add cube diagonal nodes?
        self._add_square_diagonal_nodes()
        self.side_len = self.side_len / 2

    def divide_edges(self):
        super().divide_edges()
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level + 1, self.current_level])
        self._start_new_level()


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
    where_result = np.where((dist < np.sqrt(2)*side_len) & (0 < dist)) #np.sqrt(2)*
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
               |__8___|

    First neighbours of 1: 2, 6, 3, 8
    Second neighbours of 1: 5, 7
    """
    direct_neighbours = list(graph.neighbors(node))
    # don't repeat the same second neighbour twice
    seen_seconds = []
    for neighbor_list in [graph.neighbors(n) for n in direct_neighbours]:
        for n in neighbor_list:
            if n != node and n not in direct_neighbours and n not in seen_seconds:
                seen_seconds.append(n)
                yield n


def third_neighbours(graph: nx.Graph, node):
    """
    Yield second neighbors of node in graph. Analogous to second neighbours, one more degree of separation

    Example:

        5------6
        |      |
        2 ---- 1 ---- 3 ---- 7 ---- 9
               |      |
               11__8__10

    First neighbours of 1: 2, 6, 3, 11
    Second neighbours of 1: 5, 8, 10, 7
    Third neighbours of 1: 9
    """
    direct_neighbours = list(graph.neighbors(node))
    sec_neighbours = list(second_neighbours(graph, node))
    # don't repeat seen third neighbours either
    third_seen = []
    for neighbor_list in [graph.neighbors(n) for n in sec_neighbours]:
        for n in neighbor_list:
            # conditions: n must not be: the node itself, its first or second neighbour or already seen third neighbour
            if n != node and n not in direct_neighbours and n not in sec_neighbours and n not in third_seen:
                third_seen.append(n)
                yield n


class IcosahedronPolytope(Polyhedron):
    """
    IcosahedronPolytope is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
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
                            projection=normalise_vectors(np.array(vert)))  #project_grid_on_sphere(vert[np.newaxis, :]))
        for key, value in point_connections.items():
            for vi in value:
                self.G.add_edge(key, tuple(vi),
                                length=dist_on_sphere(self.G.nodes[key]["projection"],
                                                      self.G.nodes[tuple(vi)]["projection"]))
        # just to check ...
        assert self.G.number_of_nodes() == 12
        assert self.G.number_of_edges() == 30
        for node in self.G.nodes(data=True):
            assert len(node[1]["face"]) == 5 and node[1]["level"] == 0
        self.side_len = side_len / 2

    def divide_edges(self):
        """
        Subdivide once. If previous faces are triangles, adds one point at mid-point of each edge. If they are
        squares, adds one point at mid-point of each edge + 1 in the middle of the face. New points will have a higher
        level attribute.
        """
        # this positions all new points we need - simply in the middle of previous edges
        super().divide_edges()
        # also add edges between second neighbours that form triangles
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level + 1, self.current_level + 1])
        self._start_new_level()


class Cube3DPolytope(Polyhedron):
    """
    CubeGrid is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D cube + 6 vertices at mid-faces. It is possible to subdivide the sides, in that case a new
    point always appears in the middle of a square and half of previous sides.
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
                            projection=normalise_vectors(np.array(vert)))
        for key, value in point_connections.items():
            for vi in value:
                self.G.add_edge(key, tuple(vi),
                                length=dist_on_sphere(self.G.nodes[key]["projection"],
                                                      self.G.nodes[tuple(vi)]["projection"]))
        # create diagonal nodes
        self._add_square_diagonal_nodes()
        self.side_len = self.side_len / 2

    def divide_edges(self):
        super().divide_edges()
        # also add the diagonals in the other direction
        self._add_edges_of_len(self.side_len * np.sqrt(2) / 2,
                               wished_levels=[self.current_level + 1, self.current_level + 1])
        # add the straight lines connecting old points to new
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level + 1, self.current_level])
        self._start_new_level()
