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
import seaborn as sns

from molgri.assertions import is_array_with_d_dim_r_rows_c_columns, all_row_norms_equal_k, form_square_array, form_cube, \
    two_sets_of_quaternions_equal, quaternion_in_array
from molgri.space.utils import normalise_vectors, dist_on_sphere


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

    def get_N_ordered_points(self, N: int = None, projections=True, as_quat=False):
        # can't order more points than there are
        if N > self.G.number_of_nodes():
            raise ValueError(f"Cannot order more points than there are! N={N} > {self.G.number_of_nodes()}")
        if N is None:
            N = self.G.number_of_nodes()
        result = np.zeros((N, self.d))
        current_points = list(self.G.nodes)
        # first point does not matter, just select the first point
        if projections:
            result[0] = self.G.nodes[tuple(current_points[0])]["projection"]
        else:
            result[0] = current_points[0]
        current_points.pop(0)

        # for the rest of the points, determine centrality of the subgraph with already used points removed
        current_index = 1
        while current_index < N and len(current_points):
            subgraph = self.G.subgraph(current_points)
            max_iter = np.max([100, 3*N])  # bigger graphs may need more iterations than default
            dict_centrality = nx.eigenvector_centrality(subgraph, max_iter=max_iter, tol=1.0e-3, weight="length")
            # key with largest value
            most_distant_point = max(dict_centrality, key=dict_centrality.get)
            # if as_quat, skip points that are rotationally equal to already added rotation quaternions
            if not as_quat or not quaternion_in_array(np.array(most_distant_point), result[:current_index]):
                if projections:
                    result[current_index] = self.G.nodes[tuple(most_distant_point)]["projection"]
                else:
                    result[current_index] = most_distant_point
                current_index += 1
            current_points.remove(most_distant_point)
        assert current_index == len(result), "Not enough points to be ordered," \
                                             " maybe quaternion duplicates are a problem?"
        return result

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
        Detect nodes that form a square. If you find them, add the middle point (average of the four) to nodes along
        with edges from the middle to the points that form the square.
        """
        # here save the new node along with the 4 "father nodes" so you can later build all connections
        square_nodes = detect_all_squares(self.G)
        # add new nodes and edges
        for new_neighbours in square_nodes:
            # new node is the midpoint of the four square ones
            new_node_arr = np.average(np.array(new_neighbours), axis=0)
            new_node = tuple(new_node_arr)
            # just in case repetition happens
            if new_node not in self.G.nodes:
                self.G.add_node(new_node, level=self.current_level, face=self._find_face(new_neighbours),
                                projection=normalise_vectors(np.array(new_node)))
                # since a square, only need distance to one edge
                distance = np.linalg.norm(new_node_arr - np.array(new_neighbours[0]))
                length = dist_on_sphere(self.G.nodes[new_node]["projection"],
                                        self.G.nodes[new_neighbours[0]]["projection"])[0]
                all_new_edges = [(new_node, n, {"p_dist": distance, "length": length}) for n in new_neighbours]
                self.G.add_edges_from(all_new_edges)

    def _add_cube_diagonal_nodes(self):
        """
        Detect nodes that form a cube. If you find them, add the middle point (average of the eight) to nodes along
        with edges from the middle to the points that form the square.
        """
        # here save the new node along with the 4 "father nodes" so you can later build all connections
        cube_nodes = detect_all_cubes(self.G)
        # add new nodes and edges
        for new_neighbours in cube_nodes:
            # new node is the midpoint of the 8 cube ones
            new_node_arr = np.average(np.array(new_neighbours), axis=0)
            new_node = tuple(new_node_arr)
            # just in case repetition happens
            if new_node not in self.G.nodes:
                self.G.add_node(new_node, level=self.current_level, face=self._find_face(new_neighbours),
                                projection=normalise_vectors(np.array(new_node)))
                # since a cube, only need distance to one edge
                distance = np.linalg.norm(new_node_arr - np.array(new_neighbours[0]))
                length = dist_on_sphere(self.G.nodes[new_node]["projection"],
                                        self.G.nodes[new_neighbours[0]]["projection"])[0]
                all_new_edges = [(new_node, n, {"p_dist": distance, "length": length}) for n in new_neighbours]
                self.G.add_edges_from(all_new_edges)

    def _add_edges_of_len(self, edge_len: float, wished_levels: list = None, only_seconds=True):
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
        for new_node in new_level:
            # searching only second neighbours at appropriate level
            if only_seconds:
                sec_neighbours = list(second_neighbours(self.G, new_node))
                sec_neighbours = [x for x in sec_neighbours if self.G.nodes[x]["level"] == wished_levels[1]]
            else:
                sec_neighbours = [x for x in self.G.nodes if self.G.nodes[x]["level"] == wished_levels[1]]
            for other_node in sec_neighbours:
                node_dist = np.linalg.norm(np.array(new_node)-np.array(other_node))
                # check face criterion
                if self._find_face([new_node, other_node]):
                    # check distance criterion
                    if np.isclose(node_dist, edge_len):
                        length = dist_on_sphere(self.G.nodes[new_node]["projection"],
                                                self.G.nodes[other_node]["projection"])[0]
                        # just to make sure that you don't add same edge in 2 different directions
                        if new_node < other_node:
                            self.G.add_edge(new_node, other_node, p_dist=edge_len, length=length)
                        else:
                            self.G.add_edge(other_node, new_node, p_dist=edge_len, length=length)

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
            face = self._find_face([node_vector, neighbour_vector])
            # diagonals get added twice, so this is necessary
            if new_point not in self.G.nodes:
                self.G.add_node(new_point, level=self.current_level + 1,
                                face=face,
                                projection=normalise_vectors(np.array(new_point)))
            # add the two new connections: old_point1 ---- new_point and new_point ---- old_point2
            dist = np.linalg.norm(np.array(new_point)-np.array(neighbour_vector))  # the second distance the same
            length = dist_on_sphere(self.G.nodes[new_point]["projection"],
                                    self.G.nodes[node_vector]["projection"])[0]
            self.G.add_edge(new_point, neighbour_vector, p_dist=dist, length=length)
            self.G.add_edge(new_point, node_vector, p_dist=dist, length=length)
            # remove the old connection: old_point1 -------- old_point2
            self.G.remove_edge(node_vector, neighbour_vector)

    def _remove_edges_of_len(self, k: float):
        """
        Remove all edges from self.G that have the length k (or close to k if float)
        """
        to_remove = []
        for edge in self.G.edges(data=True):
            n1, n2, data = edge
            if np.isclose(data["p_dist"], k):
                to_remove.append((n1, n2))
        self.G.remove_edges_from(to_remove)


class Polyhedron(Polytope, ABC):

    def __init__(self):
        """
        Polyhedron is a polytope of exactly three dimensions. The benefit: it can be plotted.
        """
        super().__init__(d=3)

    def plot_neighbours(self, ax: Axes3D, node_i=0):
        all_nodes = self.get_node_coordinates()
        node = tuple(all_nodes[node_i])
        neig = self.G.neighbors(node)
        sec_neig = list(second_neighbours(self.G, node))
        third_neig = list(third_neighbours(self.G, node))
        for sel_node in all_nodes:
            #ax.scatter(*sel_node, color="black", s=30)
            if np.allclose(sel_node, node):
                ax.scatter(*sel_node, color="red", s=40, alpha=0.5)
            if tuple(sel_node) in neig:
                ax.scatter(*sel_node, color="blue", s=38, alpha=0.5)
            if tuple(sel_node) in sec_neig:
                ax.scatter(*sel_node, color="green", s=35, alpha=0.5)
            if tuple(sel_node) in third_neig:
                ax.scatter(*sel_node, color="orange", s=32, alpha=0.5)

    def plot_points(self, ax: Axes3D, select_faces: set = None, projection: bool = False, color_by="level"):
        """
        Plot the points of the polytope + possible division points. Colored by level at which the point was added.
        Possible to select only one or a few faces on which points are to be plotted for clarity.

        Args:
            ax: axis
            select_faces: a set of face numbers that can range from 0 to number of faces of the polyhedron, e.g. {0, 5}.
                          If None, all faces are shown.
            projection: True if you want to plot the projected points, not the ones on surfaces of polytope
            color_by: "level" or "index"
        """
        level_color = ["black", "red", "blue", "green"]
        index_palette = sns.color_palette("coolwarm", n_colors=self.G.number_of_nodes())

        for i, point in enumerate(self.get_N_ordered_points(projections=projection)):  #self.get_node_coordinates()
            # select only points that belong to at least one of the chosen select_faces (or plot all if None selection)
            node = self.G.nodes[tuple(point)]
            point_faces = set(node["face"])
            point_level = node["level"]
            point_projection = node["projection"]
            if select_faces is None or len(point_faces.intersection(select_faces)) > 0:
                # color selected based on the level of the node
                if color_by == "level":
                    color = level_color[point_level]
                elif color_by == "index":
                    color = index_palette[i]
                else:
                    raise ValueError(f"The argument color_by={color_by} not possible (try 'index', 'level')")

                if projection:
                    ax.scatter(*point_projection, color=color, s=30)
                else:
                    ax.scatter(*point, color=color, s=30)

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
        for edge in self.G.edges(data=True):
            faces_edge_1 = set(self.G.nodes[edge[0]]["face"])
            faces_edge_2 = set(self.G.nodes[edge[1]]["face"])
            # both the start and the end point of the edge must belong to one of the selected faces
            if select_faces is None or \
                (len(faces_edge_1.intersection(select_faces)) > 0 and len(faces_edge_2.intersection(select_faces)) > 0):
                # usually you only want to plot edges used in division
                ax.plot(*np.array(edge[:2]).T, color="black",  **kwargs)


class PolyhedronFromG(Polyhedron):

    def __init__(self, G: nx.Graph):
        super().__init__()
        self.G = G

    def _create_level0(self):
        pass

    def divide_edges(self):
        pass


class Cube4DPolytope(Polytope):
    """
    CubeGrid is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D cube. It is possible to subdivide the sides, in that case a new point always appears in the
    middle of a square and half of previous sides.
    """

    def __init__(self):
        super().__init__(d=4)

    def _create_level0(self):
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
            self.G.add_node(tuple(vert), level=self.current_level, face=set(),
                            projection=normalise_vectors(np.array(vert)))
        for key, value in point_connections.items():
            for vi in value:
                length = dist_on_sphere(self.G.nodes[key]["projection"],
                                        self.G.nodes[tuple(vi)]["projection"])[0]
                self.G.add_edge(key, tuple(vi), p_dist=self.side_len, length=length)
        # detect faces and cells
        cells = detect_all_cubes(self.G)
        for i, cell in enumerate(cells):
            # find the indices of the vertices that construct this square in the list of vertices
            for vertex in cell:
                self.G.nodes[tuple(vertex)]["face"].add(i)
        # create cube and square diagonal nodes
        self._add_cube_diagonal_nodes()
        self._add_square_diagonal_nodes()
        self.side_len = self.side_len / 2

    def draw_one_cell(self, ax, cell_index=0, draw_edges=True):
        nodes = (
            node
            for node, data
            in self.G.nodes(data=True)
            if cell_index in data.get('face')
        )
        subgraph = self.G.subgraph(nodes)
        # find the component corresponding to the constant 4th dimension
        arnodes = np.array(subgraph.nodes)
        dim_to_keep = list(np.where(~np.all(arnodes == arnodes[0, :], axis=0))[0])
        new_nodes = {old: (old[dim_to_keep[0]], old[dim_to_keep[1]], old[dim_to_keep[2]]) for old in subgraph.nodes}
        subgraph_3D = nx.relabel_nodes(subgraph, new_nodes)
        sub_polyhedron = PolyhedronFromG(subgraph_3D)
        sub_polyhedron.plot_points(ax)
        if draw_edges:
            sub_polyhedron.plot_edges(ax)

    def divide_edges(self):
        # this is the recipe for full division of cells - important that it is before the super call!
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=True)
        self._add_edges_of_len(self.side_len*np.sqrt(2), wished_levels=[self.current_level, self.current_level],
                               only_seconds=True)
        super().divide_edges()
        self._start_new_level()


class IcosahedronPolytope(Polyhedron):
    """
    IcosahedronPolytope is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D icosahedron. It is possible to subdivide the sides, in that case a new point always appears in the
    middle of each triangle side.
    """

    def _create_level0(self):
        # DO NOT change order of points - faces will be wrong!
        # each face contains numbers - indices of vertices that border on this face
        faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                 [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
                 [6, 2, 10], [8, 6, 7], [9, 8, 1]]
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
            set_of_faces = set(faces_i for faces_i, face in enumerate(faces) if i in face)
            self.G.add_node(tuple(vert), level=self.current_level, face=set_of_faces,
                            projection=normalise_vectors(np.array(vert)))  #project_grid_on_sphere(vert[np.newaxis, :]))
        for key, value in point_connections.items():
            for vi in value:
                length = dist_on_sphere(self.G.nodes[key]["projection"],
                                        self.G.nodes[tuple(vi)]["projection"])[0]
                self.G.add_edge(key, tuple(vi), p_dist=self.side_len, length=length)
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
        faces = [[0, 1, 2, 4], [0, 2, 3, 6], [0, 1, 3, 5], [3, 5, 6, 7], [1, 4, 5, 7], [2, 4, 6, 7]]
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
            set_of_faces = tuple(faces_i for faces_i, face in enumerate(faces) if i in face)
            self.G.add_node(tuple(vert), level=self.current_level, face=set_of_faces,
                            projection=normalise_vectors(np.array(vert)))
        for key, value in point_connections.items():
            for vi in value:
                length = dist_on_sphere(self.G.nodes[key]["projection"],
                                        self.G.nodes[tuple(vi)]["projection"])[0]
                self.G.add_edge(key, tuple(vi), p_dist=self.side_len, length=length)
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

#######################################################################################################################
#                                          GRAPH HELPER FUNCTIONS
#######################################################################################################################


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
                # if no edge there, put it there just to record the distance
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


def detect_all_squares(graph: nx.Graph) -> list:
    """
    Each node of the graph is represented by 3- or 4-dimensional coordinates, meaning that distances and even angles
    to other nodes  can be calculated. Squares can be formed by a combination of: a node, 2 neighbours and 1 second
    neighbour. Detect all such occurrences and return a list in which each row is a new set of 4 points forming a
    square.

    Args:
        graph: Graph in which each node is represented by a tuple of numbers representing coordinates

    Returns:
        a list, each item a set of 4 vertices that are guaranteed to form a square
    """
    square_nodes = []
    for node in graph.nodes:
        neighbours = list(graph.neighbors(node))
        sec_neighbours = list(second_neighbours(graph, node))

        # prepare all combinations
        # a possibility for a square: node, two neighbours, one second neighbour
        for sn in sec_neighbours:
            n_choices = list(combinations(neighbours, 2))
            for two_n in n_choices:
                n1, n2 = two_n
                # points sorted by size so that no duplicates occur
                points = sorted([node, sn, n1, n2])
                if form_square_array(np.array(points)) and points not in square_nodes:
                    square_nodes.append([*points])
    return square_nodes


def detect_all_cubes(graph: nx.Graph) -> list:
    """
    See detect_all_squares, but now a cube consists of: a node, 3 direct neighbours, 3 second neighbours, 1 third
    neighbour.

    Args:
        graph: Graph in which each node is represented by a tuple of numbers representing coordinates

    Returns:
        a list, each item a set of 8 vertices that are guaranteed to form a cube
    """
    cube_nodes = []
    for node in graph.nodes:
        neighbours = list(graph.neighbors(node))
        sec_neighbours = list(second_neighbours(graph, node))
        th_neighbours = list(third_neighbours(graph, node))
        # prepare all combinations: a node, 3 direct neighbours, 3 second neighbours, 1 third neighbour
        first_choices = list(combinations(neighbours, 3))
        second_choices = list(combinations(sec_neighbours, 3))
        for tn in th_neighbours:
            for fn in first_choices:
                for sn in second_choices:
                    # points sorted by size so that no duplicates occur
                    points = sorted([node, *fn, *sn, tn])
                    array_points = np.array(points)
                    if form_cube(array_points) and points not in cube_nodes:
                        cube_nodes.append([*points])
    return cube_nodes


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cube3D = Cube3DPolytope()
    #cube3D.divide_edges()
    fig, ax = plt.subplots(1, 1, subplot_kw={
        "projection": "3d"})
    cube3D.plot_points(ax, color_by="index", projection=False)
    #cub.plot_edges(ax)
    plt.show()
    #
    # cube4D.divide_edges()
    # fig, ax = plt.subplots(1, 1, subplot_kw={
    #     "projection": "3d"})
    # cube4D.draw_one_cell(ax, draw_edges=False, cell_index=0)
    # #cub.plot_edges(ax)
    # plt.show()