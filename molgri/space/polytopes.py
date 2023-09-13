"""
NetworkX graphs representing Platonic solids in 3- or 4D.

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
from typing import Hashable, Iterable

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi, golden
from scipy.spatial.distance import cdist

from molgri.assertions import is_array_with_d_dim_r_rows_c_columns, all_row_norms_equal_k, form_square_array, form_cube
from molgri.space.utils import angle_between_vectors, normalise_vectors, dist_on_sphere, unique_quaternion_set


class Polytope(ABC):

    """
    A polytope is a d-dim object consisting of a set of nodes (vertices) and connections between them (edges) saved
    in self.G (graph).

    The basic polytope will be created when the object is initiated. All further divisions should be performed
    with the function self.divide_edges()
    """

    def __init__(self, d: int = 3, is_quat: bool = False):
        """
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
        nx.set_node_attributes(self.G, {node: i for i, node in enumerate(self.G.nodes)}, name="central_index")

    def __str__(self):
        return "Polytope"

    @abstractmethod
    def _create_level0(self):
        """This is implemented by each subclass since they have different edges, vertices and faces"""

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
        result = []
        # first point does not matter, just select the first point
        # if projections:
        #     result[0] = self.G.nodes[tuple(current_points[0])]["projection"]
        # else:
        #     result[0] = current_points[0]


        # TODO: select all points from lower levels and then random ones
        indices_to_remove = []
        for i, point in enumerate(current_points):
            node = self.G.nodes[point]
            if node["level"] < self.current_level - 1:
                indices_to_remove.append(i)
                if projections:
                    result.append(list(node["projection"]))
                else:
                    result.append(list(point))

        if len(result) > N:
            raise ValueError("You subdivided the polyhedron more times than necessary!")

        # remove points already used
        for index in sorted(indices_to_remove, reverse=True):
            del current_points[index]

        # random remaining points
        # define a seed to have repeatable results
        np.random.seed(0)
        np.random.shuffle(current_points)
        random_points = current_points[:N - len(result)]

        for point in random_points:
            if projections:
                result.append(list(self.G.nodes[point]["projection"]))
            else:
                result.append(list(point))

        result = np.array(result)
        return result

    def get_valid_graph(self, N_ordered_points: NDArray):
        my_nodes = []
        # need to calculate valid graph by dropping the points that are not within N ordered points
        valid_G = self.G.copy()
        all_in = 0
        for node, data in self.G.nodes(data=True):
            is_in_list = np.any([np.allclose(data["projection"], x) for x in N_ordered_points])
            if not is_in_list:
                #remove_and_reconnect(valid_G, node)
                #if it acts like a bridge, reconnect
                if len(valid_G.edges(node)) <= 2:
                    remove_and_reconnect(valid_G, node)
                else:
                    valid_G.remove_node(node)
            else:
                # update my nodes
                my_nodes.append(node)
            all_in += is_in_list
        assert all_in == len(N_ordered_points)
        assert valid_G.number_of_nodes() == len(N_ordered_points)
        return valid_G, my_nodes


    def divide_edges(self):
        """
        Subdivide once by putting a new point at mid-point of each existing edge and replacing this sub-edge with
        two edges from the two old to the new point.

        In sub-modules, additional edges may be added before performing divisions.
        """
        self._add_mid_edge_nodes()
        self.current_level += 1
        self.side_len = self.side_len / 2
        nx.set_node_attributes(self.G, {node: i for i, node in enumerate(self.G.nodes)}, name="central_index")

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
                for n in new_neighbours:
                    # IMPORTANT: even if distances on a polyhedron are the same, distances on the sphere no longer
                    # are since the distance of points to the origin depends on where on the face of the polyhedron
                    # the newly connected points appear
                    length = dist_on_sphere(self.G.nodes[new_node]["projection"],
                                            self.G.nodes[n]["projection"])[0]
                    self.G.add_edge(new_node, n, p_dist=distance, length=length)

    def _add_edges_of_len(self, edge_len: float, wished_levels: list[int] = None, only_seconds: bool = True,
                          only_face: bool = True):
        """
        Sometimes, additional edges among already existing points need to be added in order to achieve equal
        sub-division of all surfaces.

        In order to shorten the time to search for appropriate points, it is advisable to make use of filters:
         - only_seconds will only search for connections between points that are second neighbours of each other
         - wished_levels defines at which division level the points between which the edge is created should be
         - only_face: only connections between points on the same face will be created

        Args:
            edge_len: length of edge on the polyhedron surface that is condition for adding edges
            wished_levels: at what level of division should the points that we want to connect be?
            only_seconds: if True, search only among the points that are second neighbours
            only_face: if True, search only among the points that lie on the same face
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


class PolyhedronFromG(Polytope):

    """
    A mock polyhedron created from an existing graph. No guarantee that it actually represents a polyhedron. Useful for
    side tasks like visualisations, searches ...
    """

    def __init__(self, G: nx.Graph):
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

    def __str__(self):
        return f"Cube4D up to level {self.current_level}"

    def _add_average_point(self, list_old_points: list):
        """
        A helper function to _add_square_diagonal_nodes, _add_cube_diagonal_nodes and _add_mid_edge_nodes.
        Given a list in which every item is a set of nodes, add a new point that is the average of them and also add
        new edges between the newly created points and all of the old ones.

        It adds new edges and does not delete any existing ones.

        Args:
            list_old_points: each item is a list of already existing nodes - their average should be added as
            a new point
        """
        # TODO: refactor so that this is part of Polyhedron and adding edges is an extra step
        for new_neighbours in list_old_points:
            # new node is the midpoint of the old ones
            new_node_arr = np.average(np.array(new_neighbours), axis=0)
            new_node = tuple(new_node_arr)
            # just in case repetition happens
            if new_node not in self.G.nodes:
                self.G.add_node(new_node, level=self.current_level, face=self._find_face(new_neighbours),
                                projection=normalise_vectors(np.array(new_node)))

    def _add_point_at_len(self, edge_len: float, wished_levels: list = None, only_seconds: bool = True,
                          only_face: bool = True):
        """
        Sometimes, additional edges among already existing points need to be added in order to achieve equal
        sub-division of all surfaces.

       See also Polyhedron's function add_edges_of_len
        """
        # TODO: also join with Polyhedron
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
                        self._add_average_point([[new_node, other_node]])

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
        self.side_len = self.side_len / 2
        self.current_level += 1

    def divide_edges(self):
        """Before or after dividing edges, make sure all relevant connections are present. Then perform a division of
        all connections (point in the middle, connection split in two), increase the level and halve the side_len."""
        self._add_point_at_len(2*self.side_len*np.sqrt(2), wished_levels=[self.current_level-1, self.current_level-1],
                               only_seconds=False)
        self._add_point_at_len(2 * self.side_len * np.sqrt(3),
                               wished_levels=[self.current_level - 1, self.current_level - 1],
                               only_seconds=False)
        super().divide_edges()

        self._add_edges_of_len(2*self.side_len, wished_levels=[self.current_level-1, self.current_level-1],
                               only_seconds=False)

    def select_half_of_hypercube(self, return_central_indices: bool = False) -> list:
        """
        Select only half of points in a hypercube polytope in such a manner that double coverage is eliminated.

        Args:
            return_central_indices: if True, return the property 'central_index' of selected points.
                                    if False, return non-projected nodes (tuples) of selected points.

        Returns:
            a list of elements, each of them either a node in c4_polytope.G or its central_index

        How selection is done: select a list of all nodes (each node a 4-tuple) that have non-negative
        first coordinate. Among the nodes with first coordinate equal zero, select only the ones with
        non-negative second coordinate etc.

        Points are always returned sorted in the order of increasing central_index
        """
        points = sorted(self.G.nodes(), key=lambda n: self.G.nodes[n]['central_index'])
        non_repeating_points = []
        for i in range(4):
            for point in points:
                projected_point = self.G.nodes[point]["projection"]
                if np.allclose(projected_point[:i], 0) and projected_point[i] > 0:
                    # the point is selected
                    if return_central_indices:
                        non_repeating_points.append(self.G.nodes[point]['central_index'])
                    else:
                        non_repeating_points.append(tuple(point))
        return non_repeating_points

    def get_all_cells(self, include_only=None):
        """Returns 8 sub-graphs belonging to individual cells of hyper-cube."""
        all_subpoly = []
        # add central index as property
        if include_only is None:
            include_only = list(self.G.nodes)
        for cell_index in range(8):
            nodes = (
                node
                for node, data
                in self.G.nodes(data=True)
                if cell_index in data.get('face')
                and node in include_only
            )
            subgraph = self.G.subgraph(nodes)
            # find the component corresponding to the constant 4th dimension
            if subgraph.number_of_nodes() > 0:
                arr_nodes = np.array(subgraph.nodes)
                num_dim = len(arr_nodes[0])
                dim_to_keep = list(np.where(~np.all(arr_nodes == arr_nodes[0, :], axis=0))[0])
                removed_dim = max(set(range(num_dim)).difference(set(dim_to_keep)))
                new_nodes = {old: tuple(old[d] for d in range(num_dim) if d !=removed_dim) for old in
                             subgraph.nodes}
                #new_nodes = {old: (old[dim_to_keep[0]], old[dim_to_keep[1]], old[dim_to_keep[2]]) for old in subgraph.nodes}
                subgraph_3D = nx.relabel_nodes(subgraph, new_nodes)
            else:
                subgraph_3D = subgraph
            # create a 3D polyhedron and use its plotting functions
            sub_polyhedron = PolyhedronFromG(subgraph_3D)
            all_subpoly.append(sub_polyhedron)
            # nx.draw(subgraph_3D, labels={node: data["central_index"] for node, data in subgraph_3D.nodes(data=True)}, with_labels = True)
            # import matplotlib.pyplot as plt
            # plt.show()
        return all_subpoly


    def get_polytope_neighbours(self, point_index, include_opposing_neighbours=False, only_half_of_cube=False):
        adj_matrix = self.get_polytope_adj_matrix(include_opposing_neighbours=include_opposing_neighbours,
                                                  only_half_of_cube=only_half_of_cube)
        # easy - if all points are in adj matrix, just read i-th line of adj matrix
        if not only_half_of_cube:
            return np.nonzero(adj_matrix[point_index])[0]
        else:
            # change indices because adj matrix is smaller
            available_indices = self.select_half_of_hypercube(return_central_indices=True)
            available_indices.sort()
            if point_index in available_indices:
                return np.nonzero(adj_matrix[available_indices.index(point_index)])[0]
            else:
                return []

    def get_polytope_adj_matrix(self, include_opposing_neighbours=True, only_half_of_cube=True):
        """
        Get adjacency matrix sorted by central index. Default is to use the creation graph to get adjacency
        relationship.

        Args:
            include_opposing_neighbours ():

        Returns:

        """
        adj_matrix = nx.adjacency_matrix(self.G, nodelist=sorted(self.G.nodes(),
                                         key=lambda n: self.G.nodes[n]['central_index'])).toarray()
        if include_opposing_neighbours:
            all_central_ind = [self.G.nodes[n]['central_index'] for n in self.G.nodes]
            all_central_ind.sort()

            ind2opp_index = dict()
            for n, d in self.G.nodes(data=True):
                ind = d["central_index"]
                opp_ind = find_opposing_q(ind, self.G)
                if opp_ind in all_central_ind:
                    ind2opp_index[ind] = opp_ind
            for i, line in enumerate(adj_matrix):
                for j, el in enumerate(line):
                    if el:
                        adj_matrix[i][all_central_ind.index(ind2opp_index[j])] = True
        if only_half_of_cube:
            available_indices = self.select_half_of_hypercube(return_central_indices=True)
            available_indices.sort()
            #adj_matrix = np.where(, adj_matrix, None)
            # Create a new array with the same shape as the original array
            extracted_arr = np.empty_like(adj_matrix, dtype=float)
            extracted_arr[:] = np.nan

            # Extract the specified rows and columns from the original array
            extracted_arr[available_indices, :] = adj_matrix[available_indices, :]
            extracted_arr[:, available_indices] = adj_matrix[:, available_indices]
            adj_matrix = extracted_arr

            #adj_matrix = adj_matrix[available_indices, :]
            #adj_matrix = adj_matrix[:, available_indices]

        return adj_matrix

    def get_cdist_matrix(self, only_half_of_cube=True):
        if only_half_of_cube:
            chosen_G = self.G.subgraph(nodes=self.select_half_of_hypercube())
        else:
            chosen_G = self.G

        projected_nodes = sorted(chosen_G.nodes, key=lambda n: chosen_G.nodes[n]['central_index'])
        projected_nodes = np.array([chosen_G.nodes[n]['projection'] for n in projected_nodes])

        def constrained_angle_distance(q1, q2):
            theta = angle_between_vectors(q1, q2)
            if (theta > pi / 2):
                theta = pi - theta
            return theta

        return cdist(projected_nodes, projected_nodes, constrained_angle_distance)


class IcosahedronPolytope(Polytope):
    """
    IcosahedronPolytope is a graph object, its central feature is self.G (networkx graph). In the beginning, each node
    is a vertex of a 3D icosahedron. It is possible to subdivide the sides, in that case a new point always appears in
    the middle of each triangle side.
    """

    def __init__(self):
        super().__init__(d=3, is_quat=False)

    def __str__(self):
        return f"Icosahedron up to level {self.current_level}"

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


class Cube3DPolytope(Polytope):
    """
    In the beginning, each node is a vertex of a 3D cube + 6 vertices at mid-faces. It is possible to subdivide
    the sides, in that case a new point always appears in the middle of a square and half of previous sides.
    """

    def __init__(self):
        super().__init__(d=3, is_quat=False)

    def __str__(self):
        return f"Cube3D up to level {self.current_level}"

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
        """
        Subdivide each existing edge. Before the division make sure there are diagonal and straight connections so that
        new points appear at mid-edges and mid-faces.
        """
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


def second_neighbours(graph: nx.Graph, node: Hashable) -> Iterable:
    """
    Yield second neighbors of node in graph. Ignore second neighbours that are also first neighbours.
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


def third_neighbours(graph: nx.Graph, node: Hashable) -> Iterable:
    """
    Yield third neighbors of node in graph. Analogous to second neighbours, one more degree of separation

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


def remove_and_reconnect(g: nx.Graph, node: tuple):
    """Remove a node and reconnect edges, adding the properties of previous edges together."""
    sources = list(g.neighbors(node))
    targets = list(g.neighbors(node))

    new_edges = list(product(sources, targets))
    distances = [g.edges[(node, s)]["p_dist"] + g.edges[(node, t)]["p_dist"] for s, t in new_edges]
    # remove self-loops
    new_edges = [(edge[0], edge[1], {"p_dist": distances[j]}) for j, edge in enumerate(new_edges) if
                 edge[0] != edge[1]]
    g.add_edges_from(new_edges)
    g.remove_node(node)


def find_opposing_q(node_i, G):
    """
    Node_i is the index of one point in graph G.

    Return the index of the opposing point if it is in G, else None
    """
    all_nodes_dict = {G.nodes[n]['central_index']: G.nodes[n]['projection'] for n in G.nodes}
    projected = all_nodes_dict[node_i]
    opposing_projected = - projected.copy()
    opposing_projected = tuple(opposing_projected)
    # return the non-projected point if available
    for n, d in G.nodes(data=True):
        projected_n = d["projection"]
        if np.allclose(projected_n, opposing_projected):
            return d["central_index"]
    return None


if __name__ == "__main__":
    pass