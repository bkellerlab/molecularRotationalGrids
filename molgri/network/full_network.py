"""
A FullNetwork consists of FullNodes and edges that show neighbourhood either in rotational or translational space. A
FullNode consists of one TranslationNode and one RotationNode and therefore encodes the exact gridpoint in the 6D
space available to the rigid body.
"""
from __future__ import annotations
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from itertools import groupby

from molgri.network.rotation_network import RotationNode
from molgri.network.translation_network import TranslationNode
from molgri.network.abstract import AbstractNetwork, AbstractNode


class FullNode(AbstractNode):

    def __init__(self, translation_node: TranslationNode, rotation_node: RotationNode):
        self.translation_node = translation_node
        self.rotation_node = rotation_node
        self.universe = None

    def __str__(self):
        return f'({str(self.translation_node)}, {str(self.rotation_node)})'

    def is_boundary_to_bulk(self) -> bool:
        return self.translation_node.is_boundary_to_bulk()

    def __lt__(self, other: FullNode):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        - if both are the same, we compare the rotation index
        """
        return self.get_indices() < other.get_indices()

    def get_7d_coordinate(self) -> NDArray:
        """
        The central property of a FullNode is its 7d coordinate. This is provided as an array [x, y, z, q0, q1, q2,
        q3] where the first three elements encode position and the quaternion (last four elements) the orientation.

        Returns:
            an array of shape (7,) fully encoding the gridpoint in 6D configuration space of a rigid body
        """
        return np.concatenate((self.translation_node.coordinate, self.rotation_node.coordinate))

    def get_indices(self)-> list:
        """
        A getter for the translational and rotational index of this node, required to order the nodes correctly.

        Returns:
            a list of two indices representing sub-node indices
        """
        return [self.translation_node, self.rotation_node]


    @cached_property
    def volume(self) -> float:
        """
        Get the full volume of this node, assuming volumes of translation and rotation sub-nodes are independent.

        Returns:
            a float representing volume in A^3
        """
        return self.translation_node.volume * self.rotation_node.volume

    def hull(self) -> tuple:
        """
        Provides acess to hulls of sub-nodes. The concept of FullNode "hull" otherwise doesn't make too much sense.

        Returns:
            a tuple of two elements, each an array of hull vertices
        """
        return (self.translation_node.hull, self.rotation_node.hull)

    def apply_transform_on(self, molecular_coordinates: NDArray, weights: NDArray = None) -> NDArray:
        """
        Since the FullNode completely specifies the gridpoint, it can also transform the initial molecular
        coordinates to the structure associated with this gridpoint. This is performed in two steps: first the
        rotation, then the translation.

        Args:
            molecular_coordinates (NDArray): an array of shape (N_atoms, 3) the initial molecular coordinates where COM
                is at origin and orientation equals the reference orientation
            weights (NDArray): an array of shape (N_atoms,) the masses of all atoms so that rotation can be around COM

        Returns:
            an array of shape (N_atoms, 3) with transformed molecular coordinates - com is now at coordinate [x, y,
            z] and the quaternion [q0, q1, q2, q3] was applied to change the orientation
        """
        # first the rotation
        rotated_points = self.rotation_node.apply_transform_on(molecular_coordinates, weights=weights)
        # afterwards the translation
        translated_points = self.translation_node.apply_transform_on(rotated_points, weights=weights)
        return translated_points



class FullNetwork(AbstractNetwork):

    @cached_property
    def grid(self):
        """
        The grid contains 7D coordinates of all nodes ordered correctly. Each row contains [x, y, z, q0, q1, q2,
        q3] defining the position and orientation at this gridpoint

        Returns:
            an array of shape (N_nodes, 7)
        """
        coordinates = [node.get_7d_coordinate() for node in self.sorted_nodes]
        return np.array(coordinates)

    def _distances(self, edge_dict: dict) -> dict:
        return {edge_dict["edge_type"]: edge_dict["distance"]}

    def _surfaces(self, edge_dict: dict) -> dict:
        return {edge_dict["edge_type"]: edge_dict["surface"]}

    def _numerical_edge_type(self, edge_dict: dict) -> dict:
        return {edge_dict["edge_type"]: edge_dict["numerical_edge_type"]}

    def _vol(self, edge_dict: dict) -> dict:
        return {edge_dict["edge_type"]: edge_dict["vol"]}

    def get_translation_indices(self) -> NDArray:
        """
        The ordering of nodes looks like this: first N_rot elements are all the different rotations at the
        first position. Afterwards follow all the different rotations at the second position.

        Sometimes we are only interested in the translation index of each element in the grid, which is what this
        method provides.

        Returns:
            an array that looks like this: [0, 0 ... 0, 1, 1, ... 1, N_trans, N_trans ... N_trans] where every element
            is repeated N_rot-times.
        """
        N_translations, N_rotations = self.list_of_position_nodes.shape
        indices = np.array([np.repeat(i, N_rotations) for i in range(N_translations)], dtype=int)
        indices = indices.reshape(-1)
        return indices

    def get_rotation_indices(self) -> NDArray:
        """
        The ordering of nodes looks like this: first N_rot elements are all the different rotations at the
        first position. Afterwards follow all the different rotations at the second position.

        Sometimes we are only interested in the rotation index of each element in the grid, which is what this
        method provides.

        Returns:
            an array that looks like this: [0, 1, ... N_rot, 0, 1, ... N_rot ....] where the sequence 0,1 ... N_rot
            is repeated N_trans-times.
        """
        indices = np.array([node.rotation_node.index for node in self.sorted_nodes], dtype=int)
        return indices

    def get_surface_to_bulk(self):
        edges_to_bulk = [node.is_boundary_to_bulk() for node in self.sorted_nodes]
        #num_rotations = np.max(self.get_rotation_indices()) + 1

        all_unit_surfaces = np.array([node.translation_node.sphere.unit_voronoi_area for node in self.sorted_nodes])
        unit_surfaces_to_bulk = np.where(edges_to_bulk, all_unit_surfaces, 0.0)

        all_max_radii = np.array([node.translation_node.r.hull[-1] for node in self.sorted_nodes])
        radii_to_bulk = np.where(edges_to_bulk, all_max_radii, 0.0 )


        result = radii_to_bulk**2 * unit_surfaces_to_bulk #/ num_rotations
        return np.array(result)

    def get_volumes_to_bulk(self):
        edges_to_bulk = [node.is_boundary_to_bulk() for node in self.sorted_nodes]
        #num_rotations = np.max(self.get_rotation_indices()) + 1

        all_volumes = np.array([node.translation_node.volume for node in self.sorted_nodes])
        volumes_to_bulk = np.where(edges_to_bulk, all_volumes, 0.0)

        return np.array(volumes_to_bulk)

    @cached_property
    def list_of_position_nodes(self) -> NDArray:
        """
        The result is an array, first line are all nodes at the first position, the second all nodes at the second
        position and so on.
        """
        groups = np.array([list(g) for k, g in groupby(self.sorted_nodes, key=lambda o: o.translation_node)])
        return groups


    def get_property_per_position(self, property_name: str)-> Any:
        """
        The result is an array, first line are all nodes at the first position, the second all nodes at the second
        position and so on. But instead of returning the nodes directly we return some property of that node. To do
        this efficiently we use the numpy vectorize function.
        """
        groups = self.list_of_position_nodes
        func = np.vectorize(lambda o: o.get_node_property(property_name))
        return func(groups)


