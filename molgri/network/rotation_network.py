"""
Here we introduce RotationNode and RotationNetwork.

RotationNode has a single quaternion as a core and provides access to other properties such as the hull of this quaternion.

RotationNetwork contains all RotationNodes of the grid and through edges provides access to their relationships, eg.
which ones are neighbours and how far away they are.

In a FullNetwork, each FullNode consists of one RotationNode and one TranslationNode.
"""
from __future__ import annotations
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from molgri.network.abstract import AbstractNetwork, AbstractNode
from molgri.utils.spheres import exact_area_of_spherical_polygon
from molgri.utils.quaternions import (find_shared_quaternions,
                                      distance_between_quaternions,
                                      cut_off_constant_dimension_quat)


class RotationNode(AbstractNode):

    """
    This class is built around a single quaternion, representing one particular rotation.
    """

    def __init__(self, rotation_index: int, quaternion: NDArray, hypersphere_hull: NDArray = None,
                 hull_volume: float = None):
        self.index = rotation_index
        self.coordinate = quaternion
        self.hull = hypersphere_hull
        self.volume = hull_volume

    def hull(self) -> NDArray:
        return self.hull

    def __str__(self):
        return f'quat={self.index}'

    def __lt__(self, other: RotationNode):
        """
       We know a node is "larger" (should come later in sorting) if its index is larger.
        """
        return self.index < other.index


    def apply_transform_on(self, molecular_coordinates: NDArray, weights: NDArray = None) -> NDArray:
        """
        Appy the rotation of this node onto the rigid body defined by given molecular coordinates. If weights are not
        provided,the rotation is done around geometrical center, otherwise around the center of mass.

        Args:
            molecular_coordinates (NDArray): an array of molecular coordinates of shape (N_atoms, 3)
            weights (NDArray or None): usually given as a list of atomic weights of length N_atoms

        Returns:
            an array of molecular coordinates of shape (N_atoms, 3) after the rotation is applied
        """
        center_of_geometry = np.average(molecular_coordinates, axis=0, weights=weights)
        shifted_points = molecular_coordinates - center_of_geometry
        rot = Rotation.from_quat(self.coordinate, scalar_first=True)
        rotated_points = rot.apply(shifted_points)
        rotated_points += center_of_geometry
        return rotated_points


class RotationNetwork(AbstractNetwork):

    """
    The network that only describes rotations in the grid. We can e.g. see which rotations are neighbours by looking
    at the edges of this graph.
    """

    @cached_property
    def grid(self) -> NDArray:
        """
        Grid is here a (N_rotations, 4)-shaped array containing quaternions.
        """
        coordinates = [node.coordinate for node in self.sorted_nodes]
        return np.array(coordinates)

    def _distances(self, edge_dict: dict) -> dict:
        """
        This is an efficient way to set the function used to calculate distance. The distance is not actually
        calculated yet, this only happens when calculate_all_edge_properties is called.

        Args:
            edge_dict (dict): contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
            edge dict with a function that can calculate a float (here distance between nodes)
        """
        node1 = edge_dict["source"]
        node2 = edge_dict["target"]
        return {"rotational": distance_between_quaternions(node1.coordinate, node2.coordinate)}

    def _surfaces(self, edge_dict: dict) -> dict:
        """
        This is an efficient way to set the function used to calculate surfaces. The surface is not actually
        calculated yet, this only happens when calculate_all_edge_properties is called.

        Args:
            edge_dict (dict): contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
            edge dict with a function that can calculate a float (here surface between nodes)
        """
        node1 = edge_dict["source"]
        node2 = edge_dict["target"]
        shared_vertices = find_shared_quaternions(node1.hull, node2.hull)
        if np.linalg.matrix_rank(shared_vertices) < 4:
            lower_dim_points = cut_off_constant_dimension_quat(shared_vertices)
            area = exact_area_of_spherical_polygon(lower_dim_points)
        else:
            area = 0.0
        return  {"rotational": area}

    def _vol(self, edge_dict: dict) -> dict:
        node1 = edge_dict["source"]
        return  {"rotational": node1.volume}

    def _numerical_edge_type(self, edge_dict: dict) -> dict:
        """
        Here we set that rotational edges always have a numerical edge type 4. This has no particular meaning,
        it is just useful that each edge type has some unique number in case we want to plot adjacency (or surface,
        distance ...) matrices and differentiate them based on edge type.

        Args:
            edge_dict (dict): contains all information about a particular edge, especially source and target,
            the two nodes that the edge connects

        Returns:
            always number 4 for rotational edges.
        """
        return  {"rotational": 4}