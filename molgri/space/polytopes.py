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
from typing import Hashable, Iterable, Type, List

import networkx as nx
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.constants import pi, golden
from scipy.sparse import coo_array
from scipy.spatial.distance import cdist

from molgri.assertions import which_row_is_k
from molgri.space.utils import distance_between_quaternions, normalise_vectors, dist_on_sphere, unique_quaternion_set

class Polytope(ABC):

    """
    A polytope is a d-dim object consisting of a set of nodes (vertices) and connections between them (edges) saved
    in self.G (graph).

    The basic polytope will be created when the object is initiated. All further divisions should be performed
    with the function self.divide_edges()
    """

    def __init__(self, d: int = 3):
        """
        The  central property of a Polytope is its graph G. The nodes of G are integers that uniquely describe a node
        (also called central index). Each node has the following properties:
         - polytope_point (coordinates of the point on the surface of a polytope)
         - projection (coordinates of the point normed to length 1)
         - level (in which subdivision step was the point created)
         - face (to which sub-unit of te polytope does the point belong)

        Args:
            d: number of dimensions
        """

        self.G = nx.Graph()
        self.current_level = 0
        self.side_len = 0
        self.current_max_ci = 0
        self.d = d

        # saved objects in order to not re-construct them unnecessarily (unless the level of division has changed)
        self.current_nodes = (None, self.current_level)

        # function that depends on the object being created
        self._create_level0()


########################################################################################################################
#
#               GETTER METHODS, USER INTERFACE
#
########################################################################################################################

    def __str__(self):
        return f"Polytope up to level {self.current_level}"

    def get_nodes(self, N: int = None, projection: bool = False) -> NDArray:
        """
        The main getter of this object. Returns an array in which every row is either a polytope point or its
        projection, either for all points or up to node N. It is important that this is consistent, so that if you
        request 15 elements now and 18 elements later, the first 15 will be the same - also for each new construction
        of this object.

        Args:
            N (int): number of points to return; if None, return all available ones
            projection (bool): if True, return projections of points onto the (hyper)sphere

        Returns:
            an array of shape (N, self.d) in which every row is a point
        """
        # checking N
        N = self._check_N(N=N)

        # checking projection
        if projection:
            attribute_name = "projection"
        else:
            attribute_name = "polytope_point"

        result = self._get_attributes_array_sorted_by_index(attribute_name)[:N]
        if len(result) > 0:
            assert result.shape == (N, self.d)
        return result

    def _check_N(self, N: int = None) -> int:
        N_available = self.G.number_of_nodes()
        # can't order more points than there are
        if N is None:
            N = N_available
        if N > N_available:
            raise ValueError(f"Cannot order more points than there are! N={N} > {N_available}")
        return N

    def _get_attributes_array_sorted_by_index(self, attribute_name: str) -> NDArray:
        """
        Return an array where every row/item is an attribute. Always returns data on all nodes sorted by central index.

        Args:
            attribute_name (str): either a name of one of node attributes or 'polytope_point' if you want
            non-projected nodes themselves

        Returns:
            An array of attributes, the length of which will be the total number of nodes
        """
        # check if sorted nodes already available
        N_nodes = self.G.number_of_nodes()
        if N_nodes == 0:
            return np.array([])
        elif self.current_nodes[1] == N_nodes:
            all_sorted_nodes = self.current_nodes[0]
        else:
            all_sorted_nodes = np.array(sorted(self.G.nodes(), key=lambda n: self.G.nodes[n]['central_index']))
            # save if requested again
            self.current_nodes = all_sorted_nodes, N_nodes

        if attribute_name == "polytope_point":
            return all_sorted_nodes
        else:
            return np.array([self.G.nodes[tuple(n)][attribute_name] for n in all_sorted_nodes])

########################################################################################################################
#
#               ADJACENCY AND CLOSENESS OF POINTS
#
########################################################################################################################


    def get_neighbours_of(self, point_index, **kwargs):
        adj_matrix = self.get_polytope_adj_matrix(**kwargs)
        return np.nonzero(adj_matrix[point_index])[0]

    def get_polytope_adj_matrix(self, only_nodes: ArrayLike = None):
        # include_opposing_neighbours=True, nly_half_of_cube=True):
        """
        Get adjacency matrix sorted by central index. Default is to use the creation graph to get adjacency
        relationship.

        Args:
            include_opposing_neighbours ():

        Returns:

        """
        if only_nodes is None:
            only_nodes = self.get_nodes(projection=False)
        adj_matrix = nx.adjacency_matrix(self.G, nodelist=[tuple(n) for n in only_nodes])
        return adj_matrix

        # if include_opposing_neighbours:
        #     all_central_ind = [self.G.nodes[n]['central_index'] for n in self.G.nodes]
        #     all_central_ind.sort()
        #
        #     ind2opp_index = dict()
        #     for n, d in self.G.nodes(data=True):
        #         ind = d["central_index"]
        #         opp_ind = find_opposing_q(ind, self.G)
        #         if opp_ind in all_central_ind:
        #             ind2opp_index[ind] = opp_ind
        #     for i, line in enumerate(adj_matrix):
        #         for j, el in enumerate(line):
        #             if el:
        #                 adj_matrix[i][all_central_ind.index(ind2opp_index[j])] = True
        # if only_half_of_cube:
        #     available_indices = self.get_half_of_hypercube(return_central_indices=True)
        #     available_indices.sort()
        #     #adj_matrix = np.where(, adj_matrix, None)
        #     # Create a new array with the same shape as the original array
        #     extracted_arr = np.empty_like(adj_matrix, dtype=float)
        #     extracted_arr[:] = np.nan
        #
        #     # Extract the specified rows and columns from the original array
        #     extracted_arr[available_indices, :] = adj_matrix[available_indices, :]
        #     extracted_arr[:, available_indices] = adj_matrix[:, available_indices]
        #     adj_matrix = extracted_arr
        #
        #     #adj_matrix = adj_matrix[available_indices, :]
        #     #adj_matrix = adj_matrix[:, available_indices]



    def get_cdist_matrix(self, only_nodes: ArrayLike = None) -> NDArray:
        """
        Cdist matrix of distances on (hyper)spheres.

        Args:
            only_nodes (ArrayLike): enables you to provide a list of nodes & only those will be included in the result

        Returns:
            a symmetric array (N, N) in which every item is a (hyper)sphere distance between points
        """
        if only_nodes is None:
            chosen_G = self.G
        else:
            chosen_G = self.G.subgraph(nodes=only_nodes)

        projected_nodes = sorted(chosen_G.nodes, key=lambda n: chosen_G.nodes[n]['central_index'])
        projected_nodes = np.array([chosen_G.nodes[n]['projection'] for n in projected_nodes])

        if self.d == 3:
            method = "cos"
        elif self.d == 4:
            method = distance_between_quaternions
        else:
            raise ValueError("Must have 3 or 4 dimensions")
        return cdist(projected_nodes, projected_nodes, method)

########################################################################################################################
#
#               CREATION & SUBDIVISION METHODS
#
########################################################################################################################

    @abstractmethod
    def _create_level0(self):
        """This is implemented by each subclass since they have different edges, vertices and faces"""
        self._end_of_divison()

    def divide_edges(self):
        """
        Subdivide once by putting a new point at mid-point of each existing edge and replacing this sub-edge with
        two edges from the two old to the new point.

        In sub-modules, additional edges may be added before and/or after performing divisions.
        """
        self._add_mid_edge_nodes()
        self._end_of_divison()

    def _end_of_divison(self):
        """
        Use at the end of _create_level0 or divide edges. Two functions:
        1) increase the level, decrease side length
        2) randomly shuffle all newly created points and assign CI to them
        """
        # find all nodes created in current level of division
        # here exceptionally DON't use get_nodes since CI not assigned yet
        new_nodes = [n for n, l in self.G.nodes(data="level") if l == self.current_level]

        # shuffle them and assign CI
        np.random.seed(15)
        np.random.shuffle(new_nodes)
        for i, n in enumerate(new_nodes):
            self.G.nodes[tuple(n)]["central_index"] = self.current_max_ci + i
        self.current_max_ci += len(new_nodes)

        # adapt current level and side_len
        self.current_level += 1
        self.side_len = self.side_len / 2

    def _add_mid_edge_nodes(self):
        """
        For each edge in the system, add a new point in the middle of the edge and replace the previous long edge with
        two shorter ones.
        """
        indices_to_add = list(self.G.edges())
        self._add_average_point_and_edges(indices_to_add, add_edges=True)
        # remove the old connection: old_point1 -------- old_point2
        for old1, old2 in indices_to_add:
            self.G.remove_edge(old1, old2)

    def _add_polytope_point(self, polytope_point: ArrayLike, face: set = None, face_neighbours_indices=None):
        """
        This is the only method that adds nodes to self.G.

        Args:
            polytope_point (ArrayLike): coordinates on the surface of polyhedron
            face (set or None): a set of integers that assign this point to one or more faces of the polyhedron
            face_neighbours_indices (): a list of neighbours from which we can determine the face(s) of the new point (
            overriden by face if given)

        Returns:

        """
        if face_neighbours_indices is None:
            face_neighbours_indices = []
        if face is None:
            face = self._find_face(face_neighbours_indices)

        # not adding central_index yet
        self.G.add_node(tuple(polytope_point), level=self.current_level,
                        face=face, projection=normalise_vectors(polytope_point))

    def _add_edges_of_len(self, edge_len: float, wished_levels: List[int] = None, only_seconds: bool = True,
                          only_face: bool = True):
        """
        Finds and adds all possible edges of specifies length between existing nodes (optionally only between nodes
        fulfilling face/level/second neighbour condition to save computational time).

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

        # central indices of all points within wished level
        selected_level = [n for n, d in self.G.nodes(data=True) if d['level'] == wished_levels[0]]
        for new_node in selected_level:
            # searching only second neighbours of the node if this option has been selected
            if only_seconds:
                sec_neighbours = list(second_neighbours(self.G, new_node))
                sec_neighbours = [n for n in sec_neighbours if self.G.nodes[n]["level"] == wished_levels[1]]
            else:
                sec_neighbours = [n for n in self.G.nodes if self.G.nodes[n]["level"] == wished_levels[1]]
            for other_node in sec_neighbours:
                new_node_point = np.array(new_node)
                other_node_point = np.array(other_node)
                node_dist = np.linalg.norm(new_node_point-other_node_point)
                # check face criterion
                if not only_face or self._find_face([new_node, other_node]):
                    # check distance criterion
                    if np.isclose(node_dist, edge_len):
                        length = dist_on_sphere(self.G.nodes[tuple(new_node)]["projection"],
                                                self.G.nodes[tuple(other_node)]["projection"])[0]
                        self.G.add_edge(new_node, other_node, p_dist=edge_len, length=length)


    def get_N_element_graph(self, N_ordered_points: NDArray):
        #TODO
        my_nodes = []
        # need to calculate valid graph by dropping the points that are not within N ordered points
        valid_G = self.G.copy()
        all_in = 0
        nodes = sorted(self.G.nodes(), key=lambda n: self.G.nodes[n]['central_index'])
        for node in nodes:
            is_in_list = np.any([np.allclose(self.G.nodes[node]["projection"], x) for x in N_ordered_points])
            if not is_in_list:
                remove_and_reconnect(valid_G, node)
                #if it acts like a bridge, reconnect
                # if len(valid_G.edges(node)) <= 2:
                #     remove_and_reconnect(valid_G, node)
                # else:
                #     valid_G.remove_node(node)
            else:
                # update my nodes
                my_nodes.append(node)
            all_in += is_in_list

        assert all_in == len(N_ordered_points)
        assert valid_G.number_of_nodes() == len(N_ordered_points)
        return valid_G, my_nodes

    def _add_average_point_and_edges(self, list_old_indices: list, add_edges=True):
        """
        A helper function to _add_square_diagonal_nodes, _add_cube_diagonal_nodes and _add_mid_edge_nodes.
        Given a list in which every item is a set of nodes, add a new point that is the average of them and also add
        new edges between the newly created points and all of the old ones.

        It adds new edges and does not delete any existing ones.

        Args:
            list_old_indices: each item is a list of already existing nodes - the average of their polytope_points
            should be added as a new point
        """
        # indices that are currently in a sublist will be averaged and that will be a new point
        for old_points in list_old_indices:
            # new node is the midpoint of the old ones
            new_point = np.average(np.array(old_points), axis=0)
            self._add_polytope_point(new_point, face_neighbours_indices=old_points)
            if add_edges:
                for n in old_points:
                    distance = np.linalg.norm(new_point - n)
                    length = dist_on_sphere(self.G.nodes[tuple(new_point)]["projection"], self.G.nodes[tuple(n)][
                        "projection"])[0]
                    self.G.add_edge(tuple(new_point), tuple(n), p_dist=distance, length=length)

########################################################################################################################
#
#               OTHER HELP FUNCTIONS
#
########################################################################################################################

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

    def _get_count_of_point_categories(self) -> NDArray:
        """
        Helper function for testing. Check how many nodes fit in distinct "projection categories" based on how central
        they are on the hypersphere.

        Returns:
            count of nodes belonging to groups based on how far away from the unit sphere they are
        """
        return np.unique([np.linalg.norm(d["projection"] - np.array(n)) for n, d in self.G.nodes(data=True)],
                         return_counts=True)[1]

    def _get_count_of_edge_categories(self) -> NDArray:
        """
        Helper function for testing. Check how many edges fit in distinct "projection categories" based on how central
        they are on the hypersphere.

        Returns:
            count of edges belonging to groups based on how far away from the unit sphere they are
        """
        return np.unique([c for a, b, c in self.G.edges(data="p_dist")], return_counts=True)[1]

    def get_edges_of_categories(self, categories: list = None, nbunch = None, data: bool = True) -> ArrayLike:
        """
        Get only those edges that belong to categories defined by distance to the sphere.

        Args:
            categories: a list of integers; if None, use all categories
            data: whether to return a view of edges that includes data

        Returns:
            a view of edges [(n1, n2, attributes), (...), ...]
        """
        all_edges = self.G.edges(data=data)
        nbunch_edges = self.G.edges(nbunch=nbunch, data=data)

        if categories is None:
            return [e for e in all_edges if e in nbunch_edges]
        else:
            # sort according to p_dist so we can filter by indices of categories
            all_edges = sorted(all_edges, key=lambda edge: edge[2].get('p_dist', 1))
            category_counts = self._get_count_of_edge_categories()
            running_sum_counts = np.cumsum(category_counts)
            running_sum_counts = np.concatenate([[0], running_sum_counts])

            result = []

            # cant check on more categories than there are
            max_num_cat = len(category_counts)
            for cat_index in categories[:max_num_cat]:
                start, stop = running_sum_counts[cat_index:cat_index+2]
                result.extend([e for e in all_edges[start:stop] if e in nbunch_edges])
            return result


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

    def get_nodes_by_index(self, indices: list, projection=False):
        if projection:
            ci2node = {d["central_index"]:d["projection"] for n, d in self.G.nodes(data=True)}
        else:
            ci2node = {d["central_index"]: n for n, d in self.G.nodes(data=True)}
        result = []
        for i in indices:
            if i in ci2node.keys():
                result.append(ci2node[i])
        return result


class Cube4DPolytope(Polytope):
    """
    CubeGrid is a graph object, its central feature is self.G (networkx graph). In the beginning, each node is
    a vertex of a 3D cube, its center or the center of its face. It is possible to subdivide the sides, in that case
    the volumes of sub-cubes are fully sub-divided.

    Special: the attribute "face" that in 3D polyhedra actually means face here refers to the cell (3D object) to which
    this specific node belongs.
    """

    def __init__(self):
        super().__init__(d=4)

    def __str__(self):
        return f"Cube4D up to level {self.current_level}"

    def _create_level0(self):
        self.side_len = 2 * np.sqrt(1/4)
        # create vertices
        vertices = list(product((-self.side_len/2, self.side_len/2), repeat=4))
        faces = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 8, 9, 10, 11], [0, 1, 4, 5, 8, 9, 12, 13],
                 [0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15], [2, 3, 6, 7, 10, 11, 14, 15],
                 [4, 5, 6, 7, 12, 13, 14, 15], [8, 9, 10, 11, 12, 13, 14, 15]]

        assert len(vertices) == 16
        assert np.all([np.isclose(x, 0.5) or np.isclose(x, -0.5) for row in vertices for x in row])
        assert len(set(vertices)) == 16
        vertices = np.array(vertices)
        # add vertices and edges to the graph
        for i, vert in enumerate(vertices):
            belongs_to = [face_i for face_i, face in enumerate(faces) if i in face]
            self._add_polytope_point(vert, face=set(belongs_to))
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        self._add_edges_of_len(self.side_len*np.sqrt(2), wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        self._add_edges_of_len(self.side_len * np.sqrt(3), wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        super()._create_level0()

    def divide_edges(self):
        """Before or after dividing edges, make sure all relevant connections are present. Then perform a division of
        all connections (point in the middle, connection split in two), increase the level and halve the side_len."""
        super().divide_edges()
        self._add_edges_of_len(2*self.side_len, wished_levels=[self.current_level - 1, self.current_level - 1],
                               only_seconds=False)
        len_square_diagonals = 2*self.side_len*np.sqrt(2)
        self._add_edges_of_len(len_square_diagonals, wished_levels=[self.current_level-1, self.current_level-1],
                              only_seconds=False)
        len_cube_diagonals = 2 * self.side_len * np.sqrt(3)
        self._add_edges_of_len(len_cube_diagonals, wished_levels=[self.current_level - 1, self.current_level - 1],
                               only_seconds=False)

########################################################################################################################
#
#               CUBE 4D-SPECIFIC METHODS
#
########################################################################################################################

    def get_half_of_hypercube(self, projection: bool = False, N: int = None) -> NDArray:
        """
        Select only half of points in a hypercube polytope in such a manner that double coverage is eliminated.

        Args:
            projection: if True, return points projected on a hypersphere
            N: if you only want to return N points, give an integer here

        Returns:
            a list of elements, each of them either a node in c4_polytope.G or its central_index

        How selection is done: select a list of all nodes (each node a 4-tuple) that have non-negative
        first coordinate. Among the nodes with first coordinate equal zero, select only the ones with
        non-negative second coordinate etc.

        Points are always returned sorted in the order of increasing central_index
        """

        projected_points = self.get_nodes(projection=True)
        unique_projected_points = unique_quaternion_set(projected_points)

        all_ci = []
        for upp in unique_projected_points:
            all_ci.append(which_row_is_k(projected_points, upp)[0])
        all_ci.sort()

        N_available = len(all_ci)
        # can't order more points than there are
        if N is None:
            N = N_available
        if N > N_available:
            raise ValueError(f"Cannot order more points than there are! N={N} > {N_available}")

        # DO NOT use N as an argument, as you first need to select half-hypercube
        return self.get_nodes(projection=projection)[all_ci][:N]

    def get_all_cells(self, include_only: ArrayLike = None) -> List[PolyhedronFromG]:
        """
        Returns 8 sub-graphs belonging to individual cells of hyper-cube. These polytopes are re-labeled with 3D
        coordinates so that they can be plotted

        Args:
            include_only (ArrayLike): a list of nodes that should be included in result;

        Returns:
            a list, each element of a list is a 3D polyhedron corresponding to a cell in hypercube
        """
        all_subpoly = []
        if include_only is None:
            include_only = list(self.G.nodes)
        else:
            include_only = [tuple(x) for x in include_only]
        for cell_index in range(8):
            nodes = (
                node
                for node, data
                in self.G.nodes(data=True)
                if cell_index in data.get('face')
                and node in include_only
            )
            subgraph = self.G.subgraph(nodes).copy()
            # find the component corresponding to the constant 4th dimension
            if subgraph.number_of_nodes() > 0:
                arr_nodes = np.array(subgraph.nodes)
                num_dim = len(arr_nodes[0])
                dim_to_keep = list(np.where(~np.all(arr_nodes == arr_nodes[0, :], axis=0))[0])
                removed_dim = max(set(range(num_dim)).difference(set(dim_to_keep)))
                new_nodes = {old: tuple(old[d] for d in range(num_dim) if d != removed_dim) for old in
                             subgraph.nodes}
                subgraph = nx.relabel_nodes(subgraph, new_nodes)
            # create a 3D polyhedron and use its plotting functions
            sub_polyhedron = PolyhedronFromG(subgraph)
            all_subpoly.append(sub_polyhedron)
        return all_subpoly

    def get_cdist_matrix(self, only_half_of_cube: bool = True, N: int = None) -> NDArray:
        """
        Update for quaternions: can decide to get cdist matrix only for half-hypersphere.

        Args:
            only_half_of_cube (bool): select True if you want only one half of the hypersphere
            N (int): number of points you want included in the cdist matrix

        Returns:
            a symmetric array (N, N) in which every item is a (hyper)sphere distance between points
        """
        if only_half_of_cube:
            only_nodes = self.get_half_of_hypercube(N=N, projection=False)
        else:
            only_nodes = self.get_nodes(N=N, projection=False)

        return super().get_cdist_matrix(only_nodes=[tuple(n) for n in only_nodes])

    def get_polytope_adj_matrix(self, include_opposing_neighbours=True, only_half_of_cube=True):
        adj_matrix = super().get_polytope_adj_matrix().toarray()

        if include_opposing_neighbours:
            ind2opp_index = dict()
            for n, d in self.G.nodes(data=True):
                ind = d["central_index"]
                opp_n = find_opposing_q(n, self.G)
                opp_ind = self.G.nodes[opp_n]["central_index"]
                if opp_n:
                    ind2opp_index[ind] = opp_ind
            for i, line in enumerate(adj_matrix):
                for j, el in enumerate(line):
                    if el:
                        adj_matrix[i][ind2opp_index[j]] = True
        if only_half_of_cube:
            available_indices = self.get_half_of_hypercube()
            available_indices = [self.G.nodes[tuple(n)]["central_index"] for n in available_indices]
            # Create a new array with the same shape as the original array
            extracted_arr = np.empty_like(adj_matrix, dtype=float)
            extracted_arr[:] = np.nan

            # Extract the specified rows and columns from the original array
            extracted_arr[available_indices, :] = adj_matrix[available_indices, :]
            extracted_arr[:, available_indices] = adj_matrix[:, available_indices]
            adj_matrix = extracted_arr
        return coo_array(adj_matrix)

    def get_neighbours_of(self, point_index, include_opposing_neighbours=True, only_half_of_cube=True):
        adj_matrix = self.get_polytope_adj_matrix(include_opposing_neighbours=include_opposing_neighbours,
                                                  only_half_of_cube=only_half_of_cube).toarray()
        if not only_half_of_cube:
            return np.nonzero(adj_matrix[point_index])[0]
        else:
            # change indices because adj matrix is smaller
            available_nodes = self.get_half_of_hypercube()
            available_is = [self.G.nodes[tuple(n)]["central_index"] for n in available_nodes]

            if point_index in available_is:
                adj_ind = np.nonzero(adj_matrix[available_is.index(point_index)])[0]

                real_ind = []

                for el in adj_ind:
                    if el in available_is:
                        real_ind.append(available_is.index(el))
                return real_ind
            else:
                return []


class IcosahedronPolytope(Polytope):
    """
    IcosahedronPolytope is a graph object, its central feature is self.G (networkx graph). In the beginning, each node
    is a vertex of a 3D icosahedron. It is possible to subdivide the sides, in that case a new point always appears in
    the middle of each triangle side.
    """

    def __init__(self):
        super().__init__(d=3)

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
            self._add_polytope_point(vert, face=set_of_faces)
        # create edges
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        # perform end of creation
        super()._create_level0()


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
    In the beginning, each node is a vertex of a 3D cube. It is possible to subdivide
    the sides, in that case a new point always appears in the middle of a square and half of previous sides.
    """

    def __init__(self):
        super().__init__(d=3)

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
            set_of_faces = set(faces_i for faces_i, face in enumerate(faces) if i in face)
            self._add_polytope_point(polytope_point=vert, face=set_of_faces)
        # create edges
        self._add_edges_of_len(self.side_len, wished_levels=[self.current_level, self.current_level],
                               only_seconds=False, only_face=False)
        # create diagonal nodes
        self._add_edges_of_len(self.side_len * np.sqrt(2),
                               wished_levels=[self.current_level, self.current_level], only_seconds=True)
        super()._create_level0()

    def divide_edges(self):
        """
        Subdivide each existing edge. Before the division make sure there are diagonal and straight connections so that
        new points appear at mid-edges and mid-faces.
        """
        super().divide_edges()
        self._add_edges_of_len(self.side_len * 2, wished_levels=[self.current_level - 1, self.current_level - 1],
                               only_seconds=True)
        self._add_edges_of_len(self.side_len * 2 * np.sqrt(2), wished_levels=[self.current_level - 1,
                                                                              self.current_level - 1],
                               only_seconds=True)

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


def remove_and_reconnect(g: nx.Graph, node: int):
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


def find_opposing_q(node, G):
    """
    Node is one node in graph G.

    Return the node of the opposing point if it is in G, else None
    """
    all_nodes_dict = {n: G.nodes[n]['projection'] for n in G.nodes}
    projected = all_nodes_dict[node]
    opposing_projected = - projected.copy()
    opposing_projected = tuple(opposing_projected)
    # return the non-projected point if available
    for n, d in G.nodes(data=True):
        projected_n = d["projection"]
        if np.allclose(projected_n, opposing_projected):
            return n
    return None
