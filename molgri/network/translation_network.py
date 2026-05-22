"""
Here we build up to TranslationNode and TranslationNetwork.

What makes translation networks more difficult than rotation networks is the fact that three translational degrees of
freedom might need to be expressed in different ways, e.g. as x-, y- and z- subgrids (Cartesian grid, optionally with
periodic boundary conditions) or as a 2D spherical grid and a radial grid.

In a FullNetwork, each FullNode consists of one RotationNode and one TranslationNode.
"""
from __future__ import annotations

from abc import ABC
from functools import cached_property
from itertools import product

import numpy as np
from numpy.typing import NDArray

from molgri.network.abstract import AbstractNetwork, AbstractNode, find_shared_vertices
from molgri.utils.spheres import circular_sector_area, dist_on_sphere


class OneDimTranslationNode:

    """
    This is a node representing a one dimensional coordinate. Its core property, coordinate, is simply a float. The
    node can represent a x-, y-, z-coordinate or also radius.
    """

    def __init__(self, direction: str, index: int, coordinate: float, hull: tuple, is_boundary_to_bulk: bool=False) -> None:
        self.index = index
        self.name = direction
        self.coordinate = coordinate
        self.hull = hull
        self.is_boundary_to_bulk = is_boundary_to_bulk

    def __str__(self) -> str:
        return f"{self.name} grid, index: {self.index}, coordinate: {self.coordinate}"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: OneDimTranslationNode):
        """
        How do we know a node is "larger" (should come later in sorting) if it has a larger index. Periodicity
        doesn't play a role here.
        """
        return self.index < other.index

class SphericalNode:

    """
    This is a node which has a usit vector at its core. All SphericalNodes together form a network on a unit sphere.
    """

    def __init__(self, spherical_index: int, unit_vector: NDArray, unit_hull = None, area: float = None):
        self.index = spherical_index
        self.coordinate = unit_vector
        self.hull = unit_hull
        self.unit_voronoi_area = area

    def __str__(self):
        return f'Sph node {self.index}'

    def __repr__(self) -> str:
        return self.__str__()

class SphericalTranslationNode(AbstractNode):
    """
    This is the complete TranslationNode if we are using spherical parametrization of translations. It is a
    container node that contains a spherical node and a radial node.
    """

    def __init__(self, r: OneDimTranslationNode, sphere: SphericalNode):
        self.r = r
        self.sphere = sphere
        self.coordinate = self.r.coordinate * self.sphere.coordinate

    def __str__(self):
        return f'Sph. tr. node {self.get_two_indices()}'

    def __repr__(self):
        return self.__str__()

    def is_boundary_to_bulk(self) -> bool:
        return self.r.is_boundary_to_bulk

    def get_two_indices(self) -> list:
        """
        To compare two nodes first based on radial index and then based on spherical index (for sorting) we use this
        function.

        Returns:
            a list containing the two indices of sub-nodees in the right order
        """
        return [self.r.index, self.sphere.index]

    def __lt__(self, other: SphericalTranslationNode):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        """
        return self.get_two_indices() < other.get_two_indices()

    @cached_property
    def hull(self) -> list:
        """
        The hull here is simply the hull of unit sphere point scaled to the upper radius and lower radius (both radii
        are the "hull" of the radial grid.

        Returns:
            a list of indices, each representing a vertex point dividing this cell from neighbouring cells
        """
        spherical_hull = self.sphere.hull
        radial_hull = self.r.hull

        vertices = []
        # add bottom vertices
        if np.isclose(radial_hull[0], 0.0):
            vertices.append(np.zeros((1, 3)))
        else:
            vertices.append(spherical_hull * np.linalg.norm(radial_hull[0]))
        # add upper vertices
        vertices.append(spherical_hull * np.linalg.norm(radial_hull[1]))
        return vertices

    @cached_property
    def volume(self) -> float:
        """
        Volume is calculated proportional to the unit voronoi area of this cell. The volumes at upper radius and
        lower radius must be substracted to get the actual radius.
        """
        radius_smaller = self.r.hull[0]
        radius_larger = self.r.hull[1]
        # how much of the unit surface is this spherical surface
        percentage = self.sphere.unit_voronoi_area / (4 * np.pi)
        # the same percentage of the volume is this cell
        position_volume = 4 / 3 * np.pi * (radius_larger ** 3 - radius_smaller ** 3) * percentage
        return position_volume

    def apply_transform_on(self, molecular_coordinates: NDArray, weights=None) -> NDArray:
        """
        For all translation nodes, the transformation of molecular coordinates is simply vector addition. Weights are
        irrelevant but the argument is provided for consistency with rotational nodes.
        """
        return molecular_coordinates + self.coordinate


class TranslationNode(AbstractNode):
    """
    This is the complete TranslationNode if we are using cartesian parametrization of translations. It is a
    container node that contains a x, y and z node.
    """

    def __init__(self, x: OneDimTranslationNode, y: OneDimTranslationNode, z: OneDimTranslationNode):
        self.x = x
        self.y = y
        self.z = z
        self.coordinate = np.array([self.x.coordinate, self.y.coordinate, self.z.coordinate])

    def __str__(self):
        return f'({self.x.index}, {self.y.index}, {self.z.index})'

    def is_boundary_to_bulk(self) -> bool:
        return np.any([self.x.is_boundary_to_bulk, self.y.is_boundary_to_bulk, self.z.is_boundary_to_bulk])

    def get_three_indices(self) -> list:
        """
        Since for sorting we want to first compare x then y then z we need the three indices in the right order.
        """
        return [self.x.index, self.y.index, self.z.index]

    def __lt__(self, other: TranslationNode):
        """
        How do we know a node is "larger" (should come later in sorting): we compare first x, then y, then z
        """
        return self.get_three_indices() < other.get_three_indices()

    @cached_property
    def hull(self) -> NDArray:
        """
        The hull of a cartesian node is easy - simply a product of hulls for each individual dimension.
        """
        x_hull = self.x.hull
        y_hull = self.y.hull
        z_hull = self.z.hull
        all_vertices = list(product(x_hull, y_hull, z_hull))
        return np.array(all_vertices)

    @cached_property
    def volume(self) -> float:
        """
        The hull of a cartesian node is easy - simply the product of the three lengths.
        """
        side_1 = self.x.hull[1] - self.x.hull[0]
        side_2 = self.y.hull[1] - self.y.hull[0]
        side_3 = self.z.hull[1] - self.z.hull[0]
        return side_1 * side_2 * side_3

    def apply_transform_on(self, molecular_coordinates: NDArray, weights=None) -> NDArray:
        """
        For all translation nodes, the transformation of molecular coordinates is simply vector addition. Weights are
        irrelevant but the argument is provided for consistency with rotational nodes.
        """
        return molecular_coordinates + self.coordinate

class TranslationNetwork(AbstractNetwork, ABC):
    """
    This is the general form of TranslationNetwork. For practical use, see SphericalTranslationNetwork or
    CartesianTranslationNetwork.
    """

    @cached_property
    def grid(self) -> NDArray:
        """
        Here you get your 3D coordinates as a (N_points, 3) array sorted in the right way.
        """
        coordinates = [node.coordinate for node in self.sorted_nodes]
        return np.array(coordinates)

class SphericalTranslationNetwork(TranslationNetwork):
    """
    The network that only describes translations in the grid. We can e.g. see which translations are neighbours by
    looking at the edges of this graph. It implements distance and surface measures specific to spherical grids.
    """

    def _radial_distance(self, node1: SphericalTranslationNode, node2: SphericalTranslationNode) -> float:
        """
        A one possible way to calculate distance - if the points are neighbours in the radial direction,
        one above the other.

        Args:
            node1 (SphericalTranslationNode): first node of the edge
            node2 (SphericalTranslationNode): second node of the edge

        Returns:
            the distance which is simply the absolute value of the difference in radii
        """
        return np.abs(node1.r.coordinate - node2.r.coordinate)

    def _spherical_distance(self, node1: SphericalTranslationNode, node2: SphericalTranslationNode) -> float:
        """
        A one possible way to calculate distance - if the points are neighbours on the sphere so that the distance is a
        curved line.

        Args:
            node1 (SphericalTranslationNode): first node of the edge
            node2 (SphericalTranslationNode): second node of the edge

        Returns:
            the distance which is simply the arch between the two points.
        """
        return dist_on_sphere(np.array(node1.sphere.coordinate), np.array(node2.sphere.coordinate))

    def _vol(self, edge_dict: dict) -> dict:
        node1 = edge_dict["source"]
        return {"r": node1.volume, "spherical": node1.volume}

    def _distances(self, edge_dict: dict) -> dict | None:
        """
        A function assigning the right distance function based on edge type.

        Args:
            edge_dict (dict):  contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
             edge dict with a function that can calculate a float (here distance between nodes)
        """
        node1 = edge_dict["source"]
        node2 = edge_dict["target"]
        if edge_dict["edge_type"] == "r":
            return {"r": self._radial_distance(node1, node2)}
        elif edge_dict["edge_type"] == "spherical":
            return {"spherical": self._spherical_distance(node1, node2)}


    def _radial_surface(self, node1: SphericalTranslationNode, node2: SphericalTranslationNode) -> float:
        """
        A one possible way to calculate surface - if the points are neighbours in the radial direction,
        one above the other. Then, the surface is a spherical polygon at in-between distance.

        Args:
            node1 (SphericalTranslationNode): first node of the edge
            node2 (SphericalTranslationNode): second node of the edge

        Returns:
            the surface which is the unit spherical Voronoi area scaled to the right radius
        """
        unit_area = node1.sphere.unit_voronoi_area
        # scale to a radius between both layers
        in_between_radius = np.min([node1.r.hull[1], node2.r.hull[1]])
        return in_between_radius ** 2 * unit_area

    def _spherical_surface(self, node1: SphericalTranslationNode, node2: SphericalTranslationNode) -> float:
        """
        A one possible way to calculate surface - if the points are neighbours on the sphere. Then, the surface is a
        circular sector, a difference between a larger and smaller circle size.

        Args:
            node1 (SphericalTranslationNode): first node of the edge
            node2 (SphericalTranslationNode): second node of the edge

        Returns:
            the surface which is determined by a shared vertices that form a piece of a circle at smaller and larger
            radius
        """
        # looking for shared vertices to span our circle slice
        shared_upper = find_shared_vertices(node1.hull[1], node2.hull[1])
        shared_lower = find_shared_vertices(node1.hull[0], node2.hull[0])
        return circular_sector_area(shared_upper, shared_lower)

    def _surfaces(self, edge_dict: dict) -> dict | None:
        """
        A function assigning the right surface function based on edge type.

        Args:
            edge_dict (dict):  contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
             edge dict with a function that can calculate a float (here surface between nodes)
        """
        node1 = edge_dict["source"]
        node2 = edge_dict["target"]
        if edge_dict["edge_type"] == "r":
            return {"r": self._radial_surface(node1, node2)}
        elif edge_dict["edge_type"] == "spherical":
            return {"spherical": self._spherical_surface(node1, node2)}

    def _numerical_edge_type(self, edge_dict: dict) -> dict:
        """
        Here we set that edges have a numerical edge type 5 (in radial direction), or 6 (in spherical
        direction). This has no particular meaning, it is just useful that each edge type has some unique number in case
        we want to plot adjacency (or surface, distance ...) matrices and differentiate them based on edge type.

        Args:
            edge_dict (dict): contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
            an integer according to edge type
        """
        return {"r": 5, "spherical": 6}


class CartesianTranslationNetwork(TranslationNetwork):
    """
    The network that only describes translations in the grid. We can e.g. see which translations are neighbours by
    looking at the edges of this graph. It implements distance and surface measures specific to Cartesian grids.
    """

    @cached_property
    def delta_x(self) -> float:
        first_node = self.sorted_nodes[0]
        return np.abs(first_node.x.hull[1] - first_node.x.hull[0])

    @cached_property
    def delta_y(self) -> float:
        first_node = self.sorted_nodes[0]
        return np.abs(first_node.y.hull[1] - first_node.y.hull[0])

    @cached_property
    def delta_z(self) -> float:
        first_node = self.sorted_nodes[0]
        return np.abs(first_node.z.hull[1] - first_node.z.hull[0])

    def _distances(self, edge_dict: dict) -> dict:
        """
        A function assigning the right distance function based on edge type.

        Args:
            edge_dict (dict):  contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
             edge dict with a function that can calculate a float (here distance between nodes)
        """
        return {"x": self.delta_x, "y": self.delta_y, "z": self.delta_z}

    def _surfaces(self, edge_dict: dict) -> dict:
        """
        A function assigning the right surface function based on edge type.

        Args:
            edge_dict (dict):  contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
             edge dict with a function that can calculate a float (here surface between nodes)
        """
        return {"x": self.delta_y*self.delta_z, "y": self.delta_x*self.delta_z, "z": self.delta_x*self.delta_y}

    def _numerical_edge_type(self, edge_dict: dict) -> dict:
        """
        Here we set that edges have a numerical edge type 1 (in x direction), 2 (in y direction) or 3 (in z direction).
        This has no particular meaning, it is just useful that each edge type has some unique number in case we want to
        plot adjacency (or surface, distance ...) matrices and differentiate them based on edge type.

        Args:
            edge_dict (dict): contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
            an integer according to edge type
        """
        return {"x": 1, "y": 2, "z": 3}

    def _vol(self, edge_dict: dict) -> dict:
        node1 = edge_dict["source"]
        return {"x": node1.volume, "y": node1.volume, "z": node1.volume}