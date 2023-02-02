"""
Polytopes are networkX graphs representing Platonic solids in 3- or 4D.

Specifically, we implement the 3D polytopes
icosahedron and cube and the 4D polytope hypercube. As a start, vertices of the polytope are added as
nodes and edges as connections between them. For the cube, a point is also added in the middle of each face diagonal
and the edges from these face points to the four vertices are added. For the hypercube, a point is added at the center
of each of the 8 sub-cells and in the middle of all square faces, edges to these points are also added.

It's important that polytopes can also be subdivided in smaller units so that grids with larger numbers of points
can be created. For this, use divide_edges() command. 3D polytopes (polyhedra) can also be plotted - for 4D polyhedra,
each 3D cell can be plotted.

Points and their projections should always be accessed with getters and include only non-repeating rotation
quaternions for cube4D.

Objects:
    - Polytope (abstract)
    - Polyhedron (abstract, only 3D, implements plotting)
    - PolyhedronFromG (helper for testing/plotting, initiated with an existing graph)
    - Cube4DPolytope
    - IcosahedronPolytope
    - Cube3DPolytope
"""

from abc import ABC, abstractmethod
from itertools import product, combinations

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import pi, golden
from seaborn import color_palette

from molgri.assertions import is_array_with_d_dim_r_rows_c_columns, all_row_norms_equal_k, form_square_array, form_cube
from molgri.space.utils import normalise_vectors, dist_on_sphere, unique_quaternion_set


class Polytope(ABC):

    def __init__(self, d: int = 3, is_quat: bool = False):
        """
        A polytope is a d-dim object consisting of a set of nodes (vertices) and connections between them (edges) saved
        in self.G (graph).

        The basic polytope will be created when the object is initiated. All further divisions should be performed
        with the function self.divide_edges()

        Args:
            d: number of dimensions
            is_quat: does the object represent quaternions (meaning that q and -q points are treated as equal)
        """
        self.G = nx.Graph()
        self.current_level = 0
        self.side_len = 0
        self.d = d
        self.is_quat = is_quat

        # saved objects in order to not re-construct them unnecessarily (unless the level of division has changed)
        self.current_nodes = (None, self.current_level)
        self.current_projections = (None, self.current_level)

        # function that depends on the object being created
        self._create_level0()

    @abstractmethod
    def _create_level0(self):
        """This is implemented by each subclass since they have different edges, vertices and faces"""

    def plot_graph(self, with_labels=True):
        """
        Plot the networkx graph of self.G.
        """
        node_labels = {i: tuple(np.round(i, 3)) for i in self.G.nodes}
        nx.draw_networkx(self.G, pos=nx.spring_layout(self.G, weight="p_dist"), with_labels=with_labels,
                         labels=node_labels)

    def get_node_coordinates(self) -> NDArray:
        """
        Get an array in which each row represents the self.d - dimensional coordinates of the node. These are the
        points on the faces of the polytope, their norm may not be one.

        If the object is_quat, approximately half of the nodes will be returned, since for quaternions q=-q.

        Returns:
            a numpy array in which each row is one of the nodes in order of their addition to the graph (not sorted)
        """
        # check if saved available and the level of division has not changed since saving
        if self.current_nodes[0] is not None and self.current_level == self.current_nodes[1]:
            # noinspection PyTypeChecker
            return self.current_nodes[0]
        nodes = self.G.nodes
        nodes = np.array(nodes)
        if self.is_quat:
            nodes = unique_quaternion_set(nodes)
        # save nodes for further use:
        self.current_nodes = (nodes, self.current_level)
        return nodes

    def get_projection_coordinates(self) -> NDArray:
        """
        Get an array in which each row represents the self.d - dimensional coordinates of the node projected on a
        self.d-dimensional unit sphere. These points must have norm 1.

        If the object is_quat, approximately half of the nodes will be returned, since for quaternions q=-q.

        Returns:
            a numpy array in which each row is one of the projections in order of their addition to the graph
            (not sorted)
        """
        # check if saved available and the level of division has not changed since saving
        if self.current_projections[0] is not None and self.current_level == self.current_projections[1]:
            # noinspection PyTypeChecker
            return self.current_projections[0]
        projections_dict = nx.get_node_attributes(self.G, "projection")
        # important to rely on the getter because it takes care of is_quat
        nodes = self.get_node_coordinates()
        projections = np.zeros((len(nodes), self.d))
        for i, n in enumerate(nodes):
            projections[i] = projections_dict[tuple(n)]

        # assertions
        is_array_with_d_dim_r_rows_c_columns(projections, d=2, r=len(nodes), c=self.d)
        all_row_norms_equal_k(projections, 1)

        # save projections for further use:
        self.current_projections = (projections, self.current_level)
        return projections

    def get_N_ordered_points(self, N: int = None, projections: bool = True) -> NDArray:
        """
        Get nodes or their projections - this getter implements sorting and truncation at N.

        Args:
            N: wished number of points to return (if None, return all)
            projections: if True, use projections on unit sphere, if False, points on the polytope

        Returns:
            an array of size (N, self.d) featuring N nodes/projections sorted for max distance between them

        Raises:
            ValueError if N larger than the number of available unique nodes
        """
        # important to use the getter
        current_points = [tuple(point) for point in self.get_node_coordinates()]
        N_available = len(current_points)
        # can't order more points than there are
        if N is None:
            N = N_available
        if N > N_available:
            raise ValueError(f"Cannot order more points than there are! N={N} > {N_available}")
        result = np.zeros((N, self.d))
        # first point does not matter, just select the first point
        if projections:
            result[0] = self.G.nodes[tuple(current_points[0])]["projection"]
        else:
            result[0] = current_points[0]
        current_points.pop(0)

        # for the rest of the points, determine centrality of the subgraph with already used points removed
        for i in range(1, N):
            subgraph = self.G.subgraph(current_points)
            max_iter = np.max([100, 3*N_available])  # bigger graphs may need more iterations than default
            dict_centrality = nx.eigenvector_centrality(subgraph, max_iter=max_iter, tol=1.0e-3, weight="length")
            # key with largest value is the point in the center of the remaining graph
            most_distant_point = max(dict_centrality, key=dict_centrality.get)
            if projections:
                result[i] = self.G.nodes[tuple(most_distant_point)]["projection"]
            else:
                result[i] = most_distant_point
            current_points.remove(most_distant_point)
        return result

    def divide_edges(self):
        """
        Subdivide once by putting a new point at mid-point of each existing edge and replacing this sub-edge with
        two edges from the two old to the new point.

        In sub-modules, additional edges may be added before performing divisions.
        """
        self._add_mid_edge_nodes()
        self.current_level += 1
        self.side_len = self.side_len / 2

    def _add_mid_edge_nodes(self):
        """
        For each edge in the system, add a new point in the middle of the edge and replace the previous long edge with
        two shorter ones.
        """
        nodes_to_add = list(self.G.edges())
        self._add_average_point_and_edges(nodes_to_add)
        # remove the old connection: old_point1 -------- old_point2
        for old1, old2 in nodes_to_add:
            self.G.remove_edge(old1, old2)

    def _add_square_diagonal_nodes(self):
        """
        Detect nodes that form a square. If you find them, add the middle point (average of the four) to nodes along
        with edges from the middle to the points that form the square.
        """
        # here save the new node along with the 4 "father nodes" so you can later build all connections
        square_nodes = detect_all_squares(self.G)
        # add new nodes and edges
        self._add_average_point_and_edges(square_nodes)

    def _add_cube_diagonal_nodes(self):
        """
        Detect nodes that form a cube. If you find them, add the middle point (average of the eight) to nodes along
        with edges from the middle to the points that form the square.
        """
        # here save the new node along with the 4 "father nodes" so you can later build all connections
        cube_nodes = detect_all_cubes(self.G)
        # add new nodes and edges
        self._add_average_point_and_edges(cube_nodes)

    def _add_average_point_and_edges(self, list_old_points: list):
        """
        A helper function to _add_square_diagonal_nodes, _add_cube_diagonal_nodes and _add_mid_edge_nodes.
        Given a list in which every item is a set of nodes, add a new point that is the average of them and also add
        new edges between the newly created points and all of the old ones.

        It adds new edges and does not delete any existing ones.

        Args:
            list_old_points: each item is a list of already existing nodes - their average should be added as
            a new point
        """
        for new_neighbours in list_old_points:
            # new node is the midpoint of the old ones
            new_node_arr = np.average(np.array(new_neighbours), axis=0)
            new_node = tuple(new_node_arr)
            # just in case repetition happens
            if new_node not in self.G.nodes:
                self.G.add_node(new_node, level=self.current_level, face=self._find_face(new_neighbours),
                                projection=normalise_vectors(np.array(new_node)))
                # since an average, only need to calculate distances once
                distance = np.linalg.norm(new_node_arr - np.array(new_neighbours[0]))
                length = dist_on_sphere(self.G.nodes[new_node]["projection"],
                                        self.G.nodes[new_neighbours[0]]["projection"])[0]
                all_new_edges = [(new_node, n, {"p_dist": distance, "length": length}) for n in new_neighbours]
                self.G.add_edges_from(all_new_edges)

    def _add_edges_of_len(self, edge_len: float, wished_levels: list = None, only_seconds=True, only_face=True):
        """
        Sometimes, additional edges among already existing points need to be added in order to achieve equal
        sub-division of all surfaces.

        In order to shorten the time to search for appropriate points, it is advisable to make use of filters:
         - only_seconds will only search for connections between points that are second neighbours of each other
         - wished_levels defines at which division level the points between which the edge is created should be
         - only_face: only connections between points on the same face will be created

        Args:
            edge_len: length of edge on the polyhedron surface that is condition for adding edges
        """
        if wished_levels is None:
            wished_levels = [self.current_level, self.current_level]
        else:
            wished_levels.sort(reverse=True)
        assert len(wished_levels) == 2
        selected_level = [x for x, y in self.G.nodes(data=True) if y['level'] == wished_levels[0]]
        for new_node in selected_level:
            # searching only second neighbours of the node if this option has been selected
            if only_seconds:
                sec_neighbours = list(second_neighbours(self.G, new_node))
                sec_neighbours = [x for x in sec_neighbours if self.G.nodes[x]["level"] == wished_levels[1]]
            else:
                sec_neighbours = [x for x in self.G.nodes if self.G.nodes[x]["level"] == wished_levels[1]]
            for other_node in sec_neighbours:
                node_dist = np.linalg.norm(np.array(new_node)-np.array(other_node))
                # check face criterion
                if not only_face or self._find_face([new_node, other_node]):
                    # check distance criterion
                    if np.isclose(node_dist, edge_len):
                        length = dist_on_sphere(self.G.nodes[new_node]["projection"],
                                                self.G.nodes[other_node]["projection"])[0]
                        # just to make sure that you don't add same edge in 2 different directions
                        if new_node < other_node:
                            self.G.add_edge(new_node, other_node, p_dist=edge_len, length=length)
                        else:
                            self.G.add_edge(other_node, new_node, p_dist=edge_len, length=length)

    def _find_face(self, node_list: list) -> set:
        """
        Find the face that is common between all nodes in the list. (May be an empty set.)

        Args:
            node_list: a list in which each element is a tuple of coordinates that has been added to self.G as a node

        Returns:
            the set of faces (may be empty) that all nodes in this list share
        """
        face = set(self.G.nodes[node_list[0]]["face"])
        for neig in node_list[1:]:
            faces_neighbour_vector = self.G.nodes[neig]["face"]
            face = face.intersection(set(faces_neighbour_vector))
        return face


class Polyhedron(Polytope, ABC):

    def __init__(self):
        """
        Polyhedron is a polytope of exactly three dimensions. The benefit: it can be plotted.
        """
        super().__init__(d=3, is_quat=False)

    def plot_neighbours(self, ax: Axes3D, node_i=0):
        """
        Want to see which points count as neighbours, second- or third neighbours of a specific node? Use this plotting
        method.
        """
        all_nodes = self.get_node_coordinates()
        node = tuple(all_nodes[node_i])
        neig = self.G.neighbors(node)
        sec_neig = list(second_neighbours(self.G, node))
        third_neig = list(third_neighbours(self.G, node))
        for sel_node in all_nodes:
            ax.scatter(*sel_node, color="black", s=30, alpha=0.5)
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
        Or: colored by index to see how sorting works. Possible to select only one or a few faces on which points
        are to be plotted for clarity.

        Args:
            ax: axis
            select_faces: a set of face numbers that can range from 0 to number of faces of the polyhedron, e.g. {0, 5}.
                          If None, all faces are shown.
            projection: True if you want to plot the projected points, not the ones on surfaces of polytope
            color_by: "level" or "index"
        """
        level_color = ["black", "red", "blue", "green"]
        index_palette = color_palette("coolwarm", n_colors=self.G.number_of_nodes())

        for i, point in enumerate(self.get_N_ordered_points(projections=False)):
            # select only points that belong to at least one of the chosen select_faces (or plot all if None selection)
            node = self.G.nodes[tuple(point)]
            point_faces = set(node["face"])
            point_level = node["level"]
            point_projection = node["projection"]
            if select_faces is None or len(point_faces.intersection(select_faces)) > 0:
                # color selected based on the level of the node or index of the sorted nodes
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
            n1_on_face = len(faces_edge_1.intersection(select_faces)) > 0
            n2_on_face = len(faces_edge_2.intersection(select_faces)) > 0
            if select_faces is None or (n1_on_face and n2_on_face):
                # usually you only want to plot edges used in division
                ax.plot(*np.array(edge[:2]).T, color="black",  **kwargs)


class PolyhedronFromG(Polyhedron):

    def __init__(self, G: nx.Graph):
        """
        This is a mock polyhedron created from an existing graph. No guarantee that it represents a polyhedron.
        Args:
            G:
        """
        super().__init__()
        self.G = G

    def _create_level0(self):
        pass


class Cube4DPolytope(Polytope):
    """
    CubeGrid is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D cube, its center or the center of its face. It is possible to subdivide the sides, in that case
    the volumes of sub-cubes are fully sub-divided.

    Special: the attribute "face" that in 3D polyhedra actually means face here refers to the cell (3D object) to which
    this specific node belongs.
    """

    def __init__(self):
        super().__init__(d=4, is_quat=True)

    def _create_level0(self):
        self.side_len = 2 * np.sqrt(1/self.d)
        # create vertices
        vertices = list(product((-self.side_len/2, self.side_len/2), repeat=4))
        assert len(vertices) == 16
        assert np.all([np.isclose(x, 0.5) or np.isclose(x, -0.5) for row in vertices for x in row])
        assert len(set(vertices)) == 16
        vertices = np.array(vertices)
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            self.G.add_node(tuple(vert), level=self.current_level, face=set(),
                            projection=normalise_vectors(np.array(vert)))
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        # detect cells
        cells = detect_all_cubes(self.G)
        for i, cell in enumerate(cells):
            # find the indices of the vertices that construct this square in the list of vertices
            for vertex in cell:
                self.G.nodes[tuple(vertex)]["face"].add(i)
        # create cube and square diagonal nodes
        self._add_cube_diagonal_nodes()
        self._add_square_diagonal_nodes()
        self.side_len = self.side_len / 2
        self.current_level += 1

    def draw_one_cell(self, ax: Axes3D, cell_index: int = 0, draw_edges: bool = True):
        """
        Since you cannot visualise a 4D object directly, here's an option to visualise the 3D sub-cells of a 4D object.

        Args:
            ax: axis
            cell_index: index of the sub-cell to plot (in cube4D that can be 0-7)
            draw_edges: use True if you want to also draw edges, False if only points
        """
        # find the points that belong to the chosen cell_index
        nodes = (
            node
            for node, data
            in self.G.nodes(data=True)
            if cell_index in data.get('face')
        )
        subgraph = self.G.subgraph(nodes)
        # find the component corresponding to the constant 4th dimension
        arr_nodes = np.array(subgraph.nodes)
        dim_to_keep = list(np.where(~np.all(arr_nodes == arr_nodes[0, :], axis=0))[0])
        new_nodes = {old: (old[dim_to_keep[0]], old[dim_to_keep[1]], old[dim_to_keep[2]]) for old in subgraph.nodes}
        subgraph_3D = nx.relabel_nodes(subgraph, new_nodes)
        # create a 3D polyhedron and use its plotting functions
        sub_polyhedron = PolyhedronFromG(subgraph_3D)
        sub_polyhedron.plot_points(ax)
        if draw_edges:
            sub_polyhedron.plot_edges(ax)

    def divide_edges(self):
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level-1, self.current_level-1],
                               only_seconds=True)
        self._add_edges_of_len(self.side_len*np.sqrt(2), wished_levels=[self.current_level-1, self.current_level-1],
                               only_seconds=True)
        super().divide_edges()


class IcosahedronPolytope(Polyhedron):
    """
    IcosahedronPolytope is a graph object, its central feature is self.G (networkx graph). In the beginning, each node
    is a vertex of a 3D icosahedron. It is possible to subdivide the sides, in that case a new point always appears in
    the middle of each triangle side.
    """

    def _create_level0(self):
        # DO NOT change order of points - faces will be wrong!
        # each face contains numbers - indices of vertices that border on this face
        faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                 [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
                 [6, 2, 10], [8, 6, 7], [9, 8, 1]]
        self.side_len = 1 / np.sin(2 * pi / 5)
        # create vertices
        vertices = [(-1, golden, 0), (1, golden, 0), (-1, -golden, 0), (1, -golden, 0),
                    (0, -1, golden), (0, 1, golden), (0, -1, -golden), (0, 1, -golden),
                    (golden, 0, -1), (golden, 0, 1), (-golden, 0, -1), (-golden, 0, 1)]
        vertices = np.array(vertices) * self.side_len / 2
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            set_of_faces = set(faces_i for faces_i, face in enumerate(faces) if i in face)
            self.G.add_node(tuple(vert), level=self.current_level, face=set_of_faces,
                            projection=normalise_vectors(np.array(vert)))
        # create edges
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        # just to check ...
        assert self.G.number_of_nodes() == 12
        assert self.G.number_of_edges() == 30
        for node in self.G.nodes(data=True):
            assert len(node[1]["face"]) == 5 and node[1]["level"] == 0
        self.side_len = self.side_len / 2
        self.current_level += 1

    def divide_edges(self):
        """
        Subdivide once. If previous faces are triangles, adds one point at mid-point of each edge. If they are
        squares, adds one point at mid-point of each edge + 1 in the middle of the face. New points will have a higher
        level attribute.
        """
        # also add edges between second neighbours that form triangles
        self._add_edges_of_len(self.side_len*2, wished_levels=[self.current_level-1, self.current_level-1],
                               only_seconds=True)
        # this positions all new points we need - simply in the middle of previous edges
        super().divide_edges()


class Cube3DPolytope(Polyhedron):
    """
    In the beginning, each node is a vertex of a 3D cube + 6 vertices at mid-faces. It is possible to subdivide
    the sides, in that case a new point always appears in the middle of a square and half of previous sides.
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

        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            set_of_faces = tuple(faces_i for faces_i, face in enumerate(faces) if i in face)
            self.G.add_node(tuple(vert), level=self.current_level, face=set_of_faces,
                            projection=normalise_vectors(np.array(vert)))
        # create edges
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        # create diagonal nodes
        self._add_square_diagonal_nodes()
        self.side_len = self.side_len / 2
        self.current_level += 1

    def divide_edges(self):
        # also add the diagonals in the other direction
        self._add_edges_of_len(self.side_len * np.sqrt(2),
                               wished_levels=[self.current_level-1, self.current_level-1], only_seconds=True)
        # add the straight lines connecting old points to new
        if self.current_level >= 2:
            self._add_edges_of_len(self.side_len*2, wished_levels=[self.current_level-1, self.current_level-2],
                                   only_seconds=True)
        super().divide_edges()

#######################################################################################################################
#                                          GRAPH HELPER FUNCTIONS
#######################################################################################################################


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
