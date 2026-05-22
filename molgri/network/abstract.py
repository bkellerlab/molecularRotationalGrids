"""
This is the general version of nodes and networks I use to create grids. The Translation- and Rotation- -Nodes and
-Networks will inherit from these abstract objects.

We also introduce ReducedSphericalVoronoi, which is the same as scipy's SphericalVoronoi, just that no vertices are
duplicated. Any time we want to use spherical voronois in our program we should access them through the function
get_spherical_voronoi().
"""

from __future__ import annotations

from abc import abstractmethod, ABC
from copy import copy
from functools import cached_property
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.spatial import SphericalVoronoi

from molgri.constants import UNIQUE_TOL
from molgri.utils.arrays import all_rows_unique, k_argmin_in_array, which_row_is_k


class AbstractNode(ABC):

    """
    Nodes of my networks, e.g. RotationNode or TranslationNode are all objects that inherit from this one.
    """

    @abstractmethod
    def __lt__(self, other: "AbstractNode") -> bool:
        pass

    @property
    @abstractmethod
    def hull(self) -> NDArray:
        pass

    def get_node_property(self, property_name: str) -> Any:
        """
        Just a simple getter for any node properties. Useful if e.g. getting this property for the full network.

        Args:
            property_name (str): the name of the property, eg to access self.coordinate write "coordinate"

        Returns:
            the value of the property, can be almost anything
        """
        return self.__dict__[property_name]

    @abstractmethod
    def apply_transform_on(self, molecular_coordinates: NDArray, weights: NDArray = None) -> NDArray:
        """
        Gives a prescription how this node should modify the molecular coordinates. E.g. translation node will
        translate all coordinates, rotation node will rotate them around the center of mass (or geometry if weights
        not given).

        Args:
            molecular_coordinates (NDArray): array of shape (N_atoms, 3) that will be modified
            weights (NDArray): array of shape (N_atoms,) that might be needed for transformation if it relates to the
                               center of mass

        Returns:
            array of shape (N_atoms, 3) of modified coordinates
        """
        pass

class AbstractNetwork(nx.Graph, ABC):

    """
    Just a bit enhanced networkx Graph that I use for my RotationNetwork, TranslationNetwork and FullNetwork.

    Important for all networks: use self.sorted_nodes in setters and getters since the ordering of nodes should be
    respected.

    The assumptions are:
        - all nodes are objects that can be sorted (have implemented __lt__)
        - all nodes have the properties hull and volume
        - edges have properties edge_type and methods are implemented to further calculate numeric_edge_type, distance
         and surface
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if there is only one node there are no edges and therefore no edge properties
        if self.number_of_nodes() > 1:
            self.calculate_all_edge_properties()

    def create_pseudotrajectory_coordinates_from(self, moving_coordinates: NDArray, weights: NDArray = None) -> list:
        """
        This is the method that applies the effects of each node to the given coordinates in the sorted order so that we
        get a list of transformed coordinates, each one transformed by a different node.

        Args:
            moving_coordinates (NDArray): array of shape (N_atoms, 3) that will be modified by each node
            weights (NDArray): array of shape (N_atoms,) containing the masses of atoms

        Returns:
            a list of lengths (N_nodes, ), each containing a (N_atoms, 3) array transformed in the specific way,
            the order is the order of gridpoints
        """
        nodes = [node.apply_transform_on(moving_coordinates, weights=weights) for node in sorted(self.nodes)]
        return nodes

    def add_node_properties(self, sorted_values_list: list, property_name: str) -> None:
        """
        General setter of properties for all nodes in the network.

        It is important to use setters and getters because the order of nodes is very important! (Nodes in a network
        are in general not sorted).

        Args:
            sorted_values_list (list): a list of length N_nodes, each element will be set to a corresponding node
            property_name (str): the name of the property, e.g. to set self.coordinate write "coordinate"
        """
        for node_i, node in enumerate(self.sorted_nodes):
            node.__dict__[property_name] = sorted_values_list[node_i]

    def get_node_properties(self, property_name: str) -> list:
        """
        General getter of properties for all nodes in the network.

        It is important to use setters and getters because the order of nodes is very important! (Nodes in a network
        are in general not sorted).

        Args:
            property_name (str): the name of the property, e.g. to set self.coordinate write "coordinate"
        Returns:
            a list of length N_nodes, each element the property of one node
        """
        chosen_property = [node.get_node_property(property_name) for node in self.sorted_nodes]
        return chosen_property

    def get_node_indices_N_lowest_energies(self, N: int) -> NDArray:
        """
        Get the indices of N sorted_nodes that have the smallest value of property energy.

        Args:
            N (int): how many nodes should be returned

        Returns:
            an array of shape (N,) containing indices (as integers) of relevant nodes
        """
        all_energies = np.array(self.get_node_properties("energy"))
        return k_argmin_in_array(all_energies, N)


    @cached_property
    def sorted_nodes(self):
        """
        This is the property that saves nodes in the desired order (order is determined by the __lt__ method of nodes).
        """
        nodes = [node for node in sorted(self.nodes)]
        return nodes

    @cached_property
    def volumes(self) -> NDArray:
        volumes = [node.volume for node in self.sorted_nodes]
        volumes = np.array(volumes, dtype=float)
        return volumes


    @cached_property
    def hulls(self):
        hulls = [node.hull for node in self.sorted_nodes]
        return hulls

    @abstractmethod
    def grid(self) -> NDArray:
        pass

    @abstractmethod
    def _distances(self, *edge_dict) -> dict:
        """
        Must return a dict in which for every edge type a method to calculate distance is returned. Edge properties
        dictionary can be given as an argument.
        """
        pass

    @abstractmethod
    def _surfaces(self, *edge_dict) -> dict:
        """
        Must return a dict in which for every edge type a method to calculate surface is returned. Edge properties
        dictionary can be given as an argument.
        """
        pass

    @abstractmethod
    def _vol(self, *edge_dict) -> dict:
        """
        """
        pass

    @abstractmethod
    def _numerical_edge_type(self, *edge_dict) -> dict:
        """
        Must return a dict in which for every edge type a number is returned.
        """
        pass

    def calculate_all_edge_properties(self) -> None:
        """
        This is the high-level function that calculates different properties of edges. This will be run automatically
        when creating new networks, so a new desired edge property should be recorded here.
        """
        df_edges = nx.to_pandas_edgelist(self)
        # now list all properties to be calculated
        df_edges["numerical_edge_type"] = df_edges.apply(
            lambda row: self._numerical_edge_type(row.to_dict())[row["edge_type"]], axis=1)
        df_edges["distance"] = df_edges.apply(
            lambda row: self._distances(row.to_dict())[row["edge_type"]], axis=1)
        df_edges["surface"] = df_edges.apply(
            lambda row: self._surfaces(row.to_dict())[row["edge_type"]], axis=1)
        df_edges["vol"] = df_edges.apply(
            lambda row: self._vol(row.to_dict())[row["edge_type"]], axis=1)
        for attribute in ["distance", "surface", "numerical_edge_type", "vol"]:
            nx.set_edge_attributes(self, df_edges.set_index(["source", "target"])[attribute].to_dict(), name=attribute)

    @cached_property
    def adjacency_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=bool)

    @cached_property
    def adjacency_type_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=int, weight="numerical_edge_type")

    @cached_property
    def adjacency_volume(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=float, weight="vol")

    @cached_property
    def distance_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=float, weight="distance")

    @cached_property
    def surface_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=float, weight="surface")


class ReducedSphericalVoronoi(SphericalVoronoi):
    """
    This layer on top of SphericalVoronoi is only there to remove repeating vertices in the typical scipy
    SphericalVoronoi, since they can cause errors in our determination of neighbourhood and dividing areas between
    neighbours.
    """

    def __init__(self, points: NDArray, radius: float = 1.0, threshold: float = 10 ** -UNIQUE_TOL):
        assert len(points.shape) == 2, "Must provide a 2D array of points"
        self.num_dimensions = points.shape[1]
        num_points = len(points)

        if num_points <= 4:
            raise ValueError(f"You are using ReducedSphericalVoronoi for < 5 points, where you should be using MikroVoronoi.")

        super().__init__(points, radius=radius, threshold=threshold)
        if self.num_dimensions == 3:
            # this is saved in order to be accessible later since the scipy function assumes redundant vertices
            self.areas = super().calculate_areas()
        self._purge_redundant_voronoi_vertices()
        # make sure no repeated vertices now
        all_rows_unique(self.vertices)

    def calculate_areas(self) -> NDArray:
        """
        This is overwritten with previous values so that the regions are not messed up again after calculating areas.

        Returns:
            an array of areas the same length as the number of points
        """
        return self.areas

    def get_adjacency_matrix(self) -> coo_array:
        """
        Create a (N_points, N_points) boolean sparse matrix that has an entry of 0 at row i, column j (and row j
        column i) if points i and j aren't neighbours, 1 if they are.

        How is neighbourhood (adjacency) determined? Adjacent points share at least dimension-1 vertices. This means
        that two points on a sphere (3D points) are neighbours if they share at least 2 vertices and two points on a
        hypersphere (4D points - quaternions) are neighbours if they share at least 3 vertices.
        """
        if len(self.points) == 1:
            return coo_array(np.zeros((1,1)))

        num_points = len(self.points)
        rows = []
        columns = []
        elements = []

        # neighbours have at least two spherical Voronoi vertices in common
        for index_tuple in combinations(list(range(num_points)), 2):
            set_1 = set(self.regions[index_tuple[0]])
            set_2 = set(self.regions[index_tuple[1]])

            if len(set_1.intersection(set_2)) >= self.num_dimensions - 1:
                rows.extend([index_tuple[0], index_tuple[1]])
                columns.extend([index_tuple[1], index_tuple[0]])
                elements.extend([True, True])

        adj_matrix = coo_array((elements, (rows, columns)), shape=(num_points, num_points))
        return adj_matrix

    def get_hulls(self) -> list:
        """
        For each Voronoi cell we want to know which vertices limit it (form a hull). We need to overwrite scipy
        function since we have changed the number of vertices, therefore the indices change.
        """
        return [self.vertices[region] for region in self.regions]

    def _purge_redundant_voronoi_vertices(self):
        """
        The main method of this class. Only keep unique vertices. Update which vertices belong to which region.
        """
        original_vertices = copy(self.vertices)
        # correctly determines which lines to use
        indexes = np.unique(original_vertices, axis=0, return_index=True)[1]
        new_vertices = np.array([original_vertices[index] for index in sorted(indexes)])

        # regions
        # correctly assigns
        old2new = {old_i: which_row_is_k(new_vertices, old)[0] for old_i, old in enumerate(original_vertices)}
        old_regions = self.regions
        new_regions = []
        for i, region in enumerate(old_regions):
            fresh_region = []
            for j, el in enumerate(region):
                fresh_region.append(old2new[el])
            new_regions.append(list(set(fresh_region)))

        # now we can overwrite
        self.vertices = new_vertices
        self.regions = new_regions

class MikroVoronoi(ReducedSphericalVoronoi):
    """
    This is a class mocking spherical voronoi cells for very small numbers of points (when it's impossible to
    actually get Voronoi cells). Mostly useful if we have just a single rotation but we want our code to run.
    """

    def __init__(self, points, **kwargs):
        """
        Note: we don't want to cal super().__init__ since we are just mocking the class.
        """
        assert len(points.shape) == 2, "Must provide a 2D array of points"
        self.num_dimensions = points.shape[1]
        assert self.num_dimensions in [3, 4]
        self.N_points = len(points)
        self.points = points

        if self.num_dimensions  == 3:
            self.vertices = None #np.array([[0, 0, 1]])
            self.regions = None #[[0]]
            # area of unit sphere divided into  N parts
            self.areas = np.array([4 * np.pi / self.N_points] * self.N_points)
        else:
            self.points = points
            self.vertices = None #np.array([[1, 0, 0, 0], [-1, 0, 0, 0]])
            self.regions = None #[[0], [1]]
            # hyperarea of half unit hypersphere  divided into  N parts
            self.areas = np.array([np.pi ** 2 / self.N_points] * self.N_points)

    def get_adjacency_matrix(self) -> coo_array:
        """
        For such a small number of points you are neighbour with everybody except yourself.
        """
        result = np.eye(self.N_points)
        # invert 0s and 1s
        result =1 - result
        return coo_array(result)


    def get_hulls(self):
        return [None] * self.N_points
        #eturn [self.vertices[region] for region in self.regions]



def get_spherical_voronoi(points, **kwargs) -> ReducedSphericalVoronoi:
    """
    Just a little function that determines whether we get a normal ReducedSphericalVoronoi or a MikroVoronoi. This
    should be the access to all spherical Voronois we want to use in our program.

    Args:
        points (NDArray): the points that determine centers of Voronoi cells, either shape (N_points, 3) or (N_points, 4)
        **kwargs: any other arguments e.g. radius, shouldn't normally be needed.

    Returns:
        a spherical voronoi object without duplicated vertices
    """
    if len(points) <= 4 and points.shape[1] == 3:
        return MikroVoronoi(points, **kwargs)
    elif len(points) <= 8 and points.shape[1] == 4:
        return MikroVoronoi(points, **kwargs)
    else:
        return ReducedSphericalVoronoi(points, **kwargs)


def find_shared_vertices(vertices1: NDArray, vertices2: NDArray) -> NDArray:
    """
    Given coordinates of vertices around point 1 (vertices1) and coordinates of vertices around point 2 (vertices2),
    find the intersection (vertices that belong to both sets).

    Args:
        vertices1 (NDArray): array of coordinates, shape (N_1, 3)
        vertices2 (NDArray): array of coordinates, shape (N_2, 3)

    Returns:
        array of coordinates, shape (N_3, 3) where N_3 <= N_1 and N_3 <= N_2
    """

    # coordinates of points must be converted to tuples so that intersection of sets can be used
    vertices_point1 = (tuple(i) for i in vertices1)
    vertices_point2 = (tuple(i) for i in vertices2)
    border_vertices = np.array(list(set(vertices_point1).intersection(set(vertices_point2))))
    return border_vertices

