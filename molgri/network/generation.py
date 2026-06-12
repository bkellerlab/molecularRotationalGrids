"""
Usually, the full Network needs to be created from all possible combinations of sub-grids. This is what is done here
using the functionality of nx.cartesian_product.

For example, we start from 1D grids in x-, y- and z-direction and all their combinations give us a
TranslationNetwork. Afterwards, we combine each translation vector with each rotation quaternion to get a FullNetwork.
"""

from typing import Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.spatial.transform import Rotation

from molgri.network.abstract import get_spherical_voronoi
from molgri.network.full_network import FullNetwork, FullNode
from molgri.network.rotation_network import RotationNetwork, RotationNode
from molgri.network.translation_network import CartesianTranslationNetwork, OneDimTranslationNode, \
    SphericalNode, SphericalTranslationNetwork, SphericalTranslationNode, TranslationNetwork, \
    TranslationNode
from molgri.utils.quaternions import double_coverage_from_upper_quaternions, hypersphere_voronoi_cell_volumes


def get_all_rotated_diffusion_matrices(upper_quaternions: NDArray, diffusion_matrix: NDArray) -> NDArray:
    """
    The provided input is the constant translational diffusion matrix in the body-fixed frame as well as the set of all
    quaternions used to rotate the body. After the body has been rotated with some quaternion, the matrix in the
    body-fixed frame remains the same but the matrix in the
    world-fixed frame is now different. This method provides the translational diffusion matrix in the world-fixed
    frame after the rotation given by the quaternion.

    Args:
        upper_quaternions (NDArray): an array of shape (N_rot, 4) describing all rotations in our set
        diffusion_matrix (NDArray): an array of shape (3, 3) the translational diffusion matrix of the original molecule

    Returns:
         an array of shape (N_rot, 3, 3) the translational diffusion matrix after each rotation in the world-fixed frame
    """
    diffusion_matrices = []
    for quat in upper_quaternions:
        R = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        transformed_diffusion_matrix = R @ diffusion_matrix @ R.T
        diffusion_matrices.append(transformed_diffusion_matrix)
    return np.array(diffusion_matrices)

def build_quaternion_network(upper_quaternions: NDArray) -> RotationNetwork:
    """
    This is the function to build a network (including info like adjacency, volumes ...) from saved quaternions.

    Args:
        upper_quaternions (NDArray): a (N_rot, 4)-shaped array of unit quaternions

    Returns:
        a network built of RotationNode objects
    """
    G = nx.Graph()
    adj_matrix, all_hulls, all_volumes = _adjacency_hulls_from_upper_quaternions(upper_quaternions)

    # first creating nodes with their corresponding hulls
    all_layer_nodes = [RotationNode(rot_i, quat, all_hulls[rot_i], all_volumes[rot_i]) for rot_i, quat in enumerate(
        upper_quaternions)]
    G.add_nodes_from(all_layer_nodes)
    # then adding edges from the adjacency matrx
    for node_i_1, node_i_2 in zip(adj_matrix.row, adj_matrix.col):
        node1 = all_layer_nodes[node_i_1]
        node2 = all_layer_nodes[node_i_2]
        G.add_edge(node1, node2, edge_type="rotational")
    my_network = RotationNetwork(G)
    return my_network

def build_translation_network(subgrids: tuple, periodic_in) -> TranslationNetwork:
    """
    This is a high-level function that builds a translation network regardless of translation type (Cartesian or
    spherical).

    Args:
        subgrids (tuple): if the network is cartesian, there will be three elements corresponding to x-,
            y- and z-direction; if spherical, two elements corresponding to radial and unit sphere subgrids
        periodic_in (tuple): only makes sense for the Cartesian grids, a 3-element tuple that encodes whether the
            gridshould be considered periodic in x, y and/or z-direction.

    Returns:
        a TranslationNetwork object
    """
    if len(subgrids) == 2:
        # spherical grid requires 2 subgrids (radial and unit sphere)
        return _build_spherical_network(subgrids)
    elif len(subgrids) == 3:
        return _build_cartesian_network(xyz_subgrids=subgrids, periodic_in=periodic_in)
    else:
        raise KeyError(f"The subgrids don't fit a cartesian nor a spherical translation algorithm.")

def _build_spherical_network(subgrids: tuple) -> SphericalTranslationNetwork:
    """
    This function should be only accessed through build_translation_network. It builds the TranslationNetwork
    specifically for spheric coordinates.

    Args:
        subgrids (tuple): a tuple of two elements, first is the 2D unit sphere grid, the second one a 1D radial grid

    Returns:
        a TranslationNetwork object
    """
    spherical_grid, r_grid = subgrids
    spherical_grid = np.array(spherical_grid)
    r_grid = np.array(r_grid)
    # first network on a sphere
    unit_spherical_voronoi = get_spherical_voronoi(spherical_grid)
    areas = unit_spherical_voronoi.calculate_areas()
    layer_adjacency = unit_spherical_voronoi.get_adjacency_matrix()
    hulls = unit_spherical_voronoi.get_hulls()

    spherical_network = nx.Graph()
    all_layer_nodes = [SphericalNode(direction_i, coo_3d, hulls[direction_i], areas[direction_i]) for direction_i,
    coo_3d in enumerate(spherical_grid)]
    spherical_network.add_nodes_from(all_layer_nodes)
    for node_i_1, node_i_2 in zip(layer_adjacency.row, layer_adjacency.col):
        node1 = all_layer_nodes[node_i_1]
        node2 = all_layer_nodes[node_i_2]
        spherical_network.add_edge(node1, node2, edge_type="spherical")

    # then radial network
    nodes  = []
    if len(r_grid) == 1:
        delta_r = r_grid[0]
    else:
        delta_r = np.abs(r_grid[1] - r_grid[0])
    for coo_i, coo in enumerate(r_grid):
        hull = (coo - delta_r / 2, coo + delta_r / 2)
        if coo_i == np.argmax(r_grid):
            is_edge_to_bulk = True
        else:
            is_edge_to_bulk = False
        nodes.append(OneDimTranslationNode("r", coo_i, coo, hull, is_edge_to_bulk))
    radial_network  = nx.Graph()
    radial_network .add_nodes_from(nodes)
    # now add edges to these sub-graphs - this is without periodicity
    for node_1, node_2 in zip(nodes[:-1], nodes[1:]):
        radial_network .add_edge(node_1, node_2, edge_type="r")

    # combine both
    full_network = nx.cartesian_product(radial_network,spherical_network)
    mapping = {(a, b): SphericalTranslationNode(a, b) for (a, b) in full_network.nodes}
    full_network = nx.relabel_nodes(full_network, mapping)
    full_network = SphericalTranslationNetwork(full_network)
    return full_network


def _build_cartesian_network(xyz_subgrids: tuple, periodic_in: list) -> CartesianTranslationNetwork:
    """
    This function should be only accessed through build_translation_network. It builds the TranslationNetwork
    specifically for Cartesian coordinates.

    It is able to handle periodicity, e.g. if the x-direction is periodic, the first and last element are neighbours
    (otherwise not).

    Args:
        xyz_subgrids (tuple): a three-element tuple, each element is a 1D grid (in x, y and z-directions)
        periodic_in (tuple): a three-element tuple, each element tells whether this direction is periodic

    Returns:
        a TranslationNetwork object
    """
    sub_networks = []
    labels = ("x", "y", "z")
    for i in range(3):
        ith_subgrid = xyz_subgrids[i]
        if len(ith_subgrid) == 1:
            delta_coo = 1.0
        else:
            delta_coo = np.abs(ith_subgrid[1] - ith_subgrid[0])
        nodes = []
        for coo_i, coo in enumerate(ith_subgrid):
            hull = (coo - delta_coo / 2, coo + delta_coo / 2)
            # if not periodic, the largest element is the edge to bulk
            if not periodic_in[i] and coo_i == np.argmax(ith_subgrid):
                is_edge_to_bulk = True
            else:
                is_edge_to_bulk = False
            nodes.append(OneDimTranslationNode(labels[i], coo_i, coo, hull, is_edge_to_bulk))

        G = nx.Graph()
        G.add_nodes_from(nodes)
        # now add edges to these sub-graphs - this is without periodicity
        for node_1, node_2 in zip(nodes[:-1], nodes[1:]):
            G.add_edge(node_1, node_2, edge_type=labels[i])
        # if periodic - add edge between first and last element
        if periodic_in[i]:
            G.add_edge(nodes[0], nodes[-1], edge_type=labels[i])
        sub_networks.append(G)


    # now combine the sub-networks
    xy_network = nx.cartesian_product(sub_networks[0], sub_networks[1])
    full_network = nx.cartesian_product(xy_network, sub_networks[2])

    mapping = {((a, b), c): TranslationNode(a, b, c) for ((a, b), c) in full_network.nodes}
    full_network = nx.relabel_nodes(full_network, mapping)
    full_network = CartesianTranslationNetwork(full_network)
    return full_network

def _adjacency_hulls_from_upper_quaternions(upper_quaternions: NDArray) -> Tuple[coo_array, NDArray, NDArray]:
    """
    In this function we deal with two properties of quaternion networks that are affected by double coverage: the
    hulls and the neighbourhood. Both must be first determined for a hypersphere that contains not only the
    quaternions +q but also -q. Afterwards, we only use the hulls of the first half of quaternions (+q), but we also
    consider the neighbours of -q to determine the neighbourhood.

    Args:
        upper_quaternions (NDArray): an array of shape (N_quaternions, 4) that saves all quaternions of the upper
            hemisphere (+q) that we are interested in

    Returns:
        a tuple of adjacency matrix of shape (N_quaternions, N_quaternions) and a list of hull vectors of length
        N_quaternions and a list of volumes of length N_quaternions
    """
    N_upper_points = upper_quaternions.shape[0]
    double_coverage_points = double_coverage_from_upper_quaternions(upper_quaternions)
    unit_spherical_voronoi = get_spherical_voronoi(double_coverage_points)
    hulls = unit_spherical_voronoi.get_hulls()
    try:
        volumes = hypersphere_voronoi_cell_volumes(double_coverage_points)
    # for very large numbers of points, the numerical integration might fail due to memory limity
    except MemoryError:
        volumes = [None] * N_upper_points


    # why can we just use the hulls of half the points? We only use the hulls to calculate lengths/areas/volumes, so
    # we need a set of points that are in the proximity of q or in the proximity of -q, but we don't really care if
    # these points are in the upper or lower hemisphere. If we combine vertices from both q and -q then we no longer
    # have only one sregion but two and so calculating its volume just gets more complex.
    single_coverage_hulls = hulls[:N_upper_points]
    single_coverage_volumes = volumes[:N_upper_points]

    # this is the matrix double the size of what we need
    adjacency_double_coverage = unit_spherical_voronoi.get_adjacency_matrix().toarray()

    # include the adjacency of opposing neighbours

    # what is happening here: the adjacency_double_coverage matrix has four quadrants. The upper left and the bottom
    # right are the same, so the upper points that are neighbours to other upper points or the lower points that are
    # neighbours to other lower points. What is of interest now are the other two sub-matrices that show when an
    # upper point is a neighbour to a lower point or vice versa.
    # Therefore, here we are copying the positions of the True values from the upper right matrix to the upper left
    # matrix to account for additional neighbourhood relations.
    upper_left = adjacency_double_coverage[:N_upper_points, :N_upper_points]
    upper_right = adjacency_double_coverage[:N_upper_points, N_upper_points:]
    upper_left += upper_right

    # now return only upper left quadrant
    return coo_array(upper_left), single_coverage_hulls, single_coverage_volumes


def create_full_network(translation_network: TranslationNetwork, rotation_network: RotationNetwork) -> FullNetwork:
    """
    This is the last step to get a FullNetwork. We simply combine all translation node options with all rotation node
    options. Two nodes in FullNetwork are neighbours if they are either rotational neighbours (with same translation
    node) or translational neighbours (with same rotation node).

    Args:
        translation_network (TranslationNetwork): a fully generated TranslationNetwork with all translational edges
        rotation_network (RotationNetwork): a fully generated RotationNetwork with all rotational edges

    Returns:
        a FullNetwork
    """
    full_network = nx.cartesian_product(translation_network, rotation_network)
    mapping = {(trans, rot): FullNode(trans, rot) for (trans, rot) in full_network.nodes}
    full_network = nx.relabel_nodes(full_network, mapping)
    # remove adjacency to itself (in case there is only 1 rotation or only 1 translation)
    full_network.remove_edges_from(nx.selfloop_edges(full_network))
    return FullNetwork(full_network)
