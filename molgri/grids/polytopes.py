import networkx as nx
import numpy as np
from scipy.constants import golden, pi
from scipy.spatial import distance_matrix
from abc import ABC, abstractmethod

from molgri.analysis.uniformity_measure import unit_dist_on_sphere
from molgri.grids.cube_grid import project_grid_on_sphere
#from plotting.abstract_plot import AbstractPlot
from molgri.my_constants import *

np.random.seed(1)


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


# class PolytopePlot(AbstractPlot):
#
#     def __init__(self, data_name: str, num_divisions=3, **kwargs):
#         self.num_divisions = num_divisions
#         plot_type = f"polytope_{num_divisions}"
#         super().__init__(data_name, fig_path=PATH_FIG_PRES, plot_type=plot_type, style_type=["empty"], dimensions=3,
#                          **kwargs)
#
#     def _prepare_data(self) -> Polytope:
#         if self.data_name == "ico":
#             ico = IcosahedronPolytope()
#         else:
#             ico = CubePolytope()
#         for n in range(self.num_divisions):
#             ico.divide_edges()
#         return ico
#
#     def _plot_data(self, face="front", **kwargs):
#         if face == "front" and self.data_name == "ico":
#             face = {12}
#         elif face == "front":
#             face = {3}
#         ico = self._prepare_data()
#         ico.plot_points(self.ax, select_faces=face, projection=False)
#         ico.plot_edges(self.ax, select_faces=face)
#
#
# if __name__ == "__main__":
#     for num_divisions in (0, 1, 2, 3):
#         PolytopePlot("ico", num_divisions=num_divisions).create(equalize=True, elev=210, azim=90, pos_limit=0.55, neg_limit=-0.6, pad_inches=-0.6)
#         PolytopePlot("cube3D", num_divisions=num_divisions).create(equalize=True, elev=0, azim=0, pos_limit=0.7, neg_limit=-0.7, pad_inches=-0.8)