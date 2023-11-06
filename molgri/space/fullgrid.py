"""
Full discretisation of space in spherical layers.

The module fullgrid combines a linear translation grid with two spherical grids: discretisation of approach vectors
(3D sphere, orientation grid) and of internal rotations of the second body (4D half-sphere of quaternions, body grid).
The three grids are commonly referred to as t_grid, o_grid and b_grid.

Position grid is a product of t_grid and o_grid and represents a set of spherical points in 3D space that are repeated
at different radii. Based on a position grid it is possible to create a Voronoi discretisation of 3D space using
identical (up to radius) layers of Voronoi surfaces between layers of grid points and connecting them with ray points
from the origin to are vertices of Voronoi cells. Methods to calculate volumes, areas and distances between centers of
such discretisation sof position space are provided.

Objects:
 - FullGrid combines 3D sphere grid, 4D sphere grid and a translation grid
 - FullVoronoiGrid extends FullGrid with methods to evaluate distances/areas/volumes of cells
 - Point represents a single point in FullVoronoiGrid and implements helper functions like identifying vertices of cell
 - ConvergenceFullGridO provides plotting data by creating a range of FullGrids with different N of o_grid points
"""
from __future__ import annotations
from copy import copy
from typing import Callable, Tuple, Optional

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import cdist
from scipy.constants import pi
from scipy.sparse import bmat, coo_array, csc_array, diags
import pandas as pd

from molgri.constants import SMALL_NS
from molgri.space.rotobj import SphereGrid3DFactory, SphereGrid3Dim, SphereGrid4DFactory, SphereGridNDim
from molgri.space.translations import TranslationParser
from molgri.naming import GridNameParser
from molgri.paths import PATH_OUTPUT_FULL_GRIDS
from molgri.space.utils import norm_per_axis, normalise_vectors, angle_between_vectors
from molgri.wrappers import save_or_use_saved, deprecated


class FullGrid:

    """
    A combination object that enables work a combination of three grids (provided by their names)

    Args:
        b_grid_name: body rotation grid (a 4D sphere grid of quaternions used to generate orientations)
        o_grid_name: origin rotation grid (a 3D sphere grid used to create approach vectors)
        t_grid_name: translation grid (a linear grid used to determine distances to origin)
    """

    def __init__(self, b_grid_name: str, o_grid_name: str, t_grid_name: str, use_saved: bool = True):

        """
        Args:
            b_grid_name: of the form 'ico_17'
            o_grid_name: of the form 'cube4D_12'
            t_grid_name: of the form '[1, 3, 4.5]'
            use_saved: try to obtain saved data if possible
        """
        b_grid_name = GridNameParser(b_grid_name, "b")
        self.b_rotations = SphereGrid4DFactory.create(alg_name=b_grid_name.get_alg(), N=b_grid_name.get_N(),
                                                      use_saved=use_saved)
        self.position_grid = PositionGrid(o_grid_name=o_grid_name, t_grid_name=t_grid_name,
                                          use_saved=use_saved)
        self.use_saved = use_saved

    def __getattr__(self, name):
        """ Enable forwarding methods to self.position_grid, so that from FullGrid you can access all properties and
         methods of PositionGrid too."""
        return getattr(self.position_grid, name)

    def __len__(self):
        """The length of the full grid is a product of lengths of all sub-grids"""
        return self.b_rotations.get_N() * len(self.get_position_grid())

    def get_position_grid(self):
        return self.position_grid

    def get_adjacency_of_orientation_grid(self) -> coo_array:
        return self.b_rotations.get_voronoi_adjacency(only_upper=True, include_opposing_neighbours=True)

    def get_name(self) -> str:
        """Name that is appropriate for saving."""
        b_name = self.b_rotations.get_name(with_dim=False)
        return f"b_{b_name}_{self.get_position_grid().get_name()}"

    def get_body_rotations(self) -> Rotation:
        """Get a Rotation object (may encapsulate a list of rotations) from the body grid."""
        return Rotation.from_quat(self.b_rotations.get_grid_as_array())

    @save_or_use_saved
    def get_full_grid_as_array(self) -> NDArray:
        """
        Return an array of shape (n_t*n_o_n_b, 7) where for every sequential step of pt, the first 3 coordinates
        describe the position in position space, the last four give the orientation in a form of a quaternion.
        """

        # WARNING! NOT CURRENTLY THE SAME AS PT!!!!!!!!!1
        result = np.full((len(self), 7), np.nan)
        position_grid = self.position_grid.get_position_grid_as_array()
        quaternions = self.b_rotations.get_grid_as_array(only_upper=True)
        current_index = 0
        for o_rot in position_grid:
            for b_rot in quaternions:
                # coordinates are (x, y, z, q0, q1, q2, q3)
                result[current_index][:3] = o_rot
                result[current_index][3:] = b_rot
                current_index += 1
        return result


    def point2cell_position_grid(self, points_vector: NDArray) -> NDArray:
        """
        This method is used to back-map any points in 3D space to their corresponding Voronoi cell indices (of position
        grid). This means that, given an array of shape (k, 3) as points_vector, the result
        will be a vector of length k in which each item is an index of the flattened position grid. The point falls
        into the Voronoi cell associated with this index.
        """
        # determine index within a layer - the layer grid point to which the point vectors are closest
        rot_points = self.o_rotations.get_grid_as_array()
        # this automatically select the one of angles that is < pi
        angles = angle_between_vectors(points_vector, rot_points)
        indices_within_layer = np.argmin(angles, axis=1)

        # determine radii of cells
        norms = norm_per_axis(points_vector)
        layers = np.zeros((len(points_vector),))
        vor_radii = self.get_between_radii()

        # find the index of the layer to which each point belongs
        for i, norm in enumerate(norms):
            for j, vor_rad in enumerate(vor_radii):
                # because norm keeps the shape of the original array
                if norm[0] < vor_rad:
                    layers[i] = j
                    break
            else:
                layers[i] = np.NaN

        layer_len = len(rot_points)
        indices = layers * layer_len + indices_within_layer
        return indices

    def nan_free_assignments(self, points_vector: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Same as point2cell_position_grid, but remove any cells that don't belong to the grid. In this way, there are no
        NaNs in the assignment array and it can be converted to the integer type
        """
        indices = self.point2cell_position_grid(points_vector)
        valid_points = points_vector[~np.isnan(indices)]
        indices = indices[~np.isnan(indices)]
        return valid_points, indices.astype(int)

    def get_full_adjacency(self):
        full_sequence = self.get_full_grid_as_array()
        n_total = len(full_sequence)
        n_o = self.o_rotations.get_N()
        n_b = self.b_rotations.get_N()
        n_t = self.t_grid.get_N_trans()

        position_adjacency = self.position_grid.get_adjacency_of_position_grid().toarray()
        if n_b > 1:
            orientation_adjacency = self.b_rotations.get_voronoi_adjacency(only_upper=True,
                                                                           include_opposing_neighbours=True)
        else:
            orientation_adjacency = coo_array([False], shape=(1,1))

        row = []
        col = []

        for i, line in enumerate(position_adjacency):
            for j, el in enumerate(line):
                if el:
                    for k in range(n_b):
                        row.append(n_b*i+k)
                        col.append(n_b*j+k)
        same_orientation_neighbours = coo_array(([True,]*len(row), (row, col)), shape=(n_total, n_total),
                                                dtype=bool)

        # along the diagonal blocks of size n_o*n_t that are neighbours exactly if their quaternions are neighbours
        if n_t * n_o > 1:
            my_blocks = [orientation_adjacency]
            my_blocks.extend([None, ] * (n_t*n_o))
            my_blocks = my_blocks * (n_t*n_o)
            my_blocks = my_blocks[:-(n_t*n_o)]
            my_blocks = np.array(my_blocks, dtype=object)
            my_blocks = my_blocks.reshape((n_t*n_o), (n_t*n_o))
            same_position_neighbours = bmat(my_blocks, dtype=float)
        else:
            return coo_array(orientation_adjacency)
        all_neighbours = same_position_neighbours + same_orientation_neighbours
        return all_neighbours


class PositionGrid:

    def __init__(self, o_grid_name: str, t_grid_name: str, use_saved: bool = True):
        """
        This is derived from FullGrid and contains methods that are connected to position grid.
        """
        o_grid_name = GridNameParser(o_grid_name, "o")
        self.o_rotations = SphereGrid3DFactory.create(alg_name=o_grid_name.get_alg(), N=o_grid_name.get_N(),
                                                      use_saved=use_saved)
        self.o_positions = self.o_rotations.get_grid_as_array()
        self.t_grid = TranslationParser(t_grid_name)
        self.use_saved = use_saved

    def __len__(self):
        """The length of the full grid is a product of lengths of all sub-grids"""
        return self.o_rotations.get_N() * self.t_grid.get_N_trans()

    def get_t_grid(self) -> TranslationParser:
        return self.t_grid

    def get_o_grid(self) -> SphereGrid3Dim:
        return self.o_rotations

    def get_name(self) -> str:
        """Name that is appropriate for saving."""
        o_name = self.o_rotations.get_name(with_dim=False)
        return f"o_{o_name}_t_{self.t_grid.grid_hash}"

    def get_radii(self) -> NDArray:
        """
        Get the radii at which points are positioned. Result is in Angstroms.
        """
        return self.t_grid.get_trans_grid()

    def get_between_radii(self) -> NDArray:
        """
        Get the radii at which Voronoi cells of the position grid should be positioned. This should be right in-between
        two orientation point layers (except the first layer that is fully encapsulated by the first voronoi layer
        and the last one that is above the last one so that the last layer of points is right in-between the two last
        Voronoi cells

        Returns:
            an array of distances, same length as the self.get_radii array but with all distances larger than the
            corresponding point radii
        """
        radii = self.get_radii()

        # get increments to each radius, remove first one and add an extra one at the end with same distance as
        # second-to-last one
        increments = list(self.t_grid.get_increments())
        if len(increments) > 1:
            increments.pop(0)
            increments.append(increments[-1])
            increments = np.array(increments)
            increments = increments / 2
        else:
            increments = np.array(increments)

        between_radii = radii + increments
        return between_radii

    def _t_and_o_2_positions(self, o_property, t_property):
        """
        Helper function to systematically combine t_grid and o_grid. Outputs an array of len n_o*n_t, can have shape
        1 or higher depending on the property

        Args:
            property ():

        Returns:

        """
        n_t = len(t_property)
        n_o = len(o_property)

        # eg coordinates
        if len(o_property.shape) > 1:
            tiled_o = np.tile(o_property, reps=(n_t, 1))
            tiled_t = np.repeat(t_property, n_o)[:, np.newaxis]
            result = tiled_o * tiled_t
        else:
            tiled_o = np.tile(o_property, reps=n_t)
            tiled_t = np.repeat(t_property, n_o)[np.newaxis, :]
            result = (tiled_o * tiled_t)[0]
        assert len(result) == n_o*n_t
        return result

    @save_or_use_saved
    def get_position_grid_as_array(self) -> NDArray:
        """
        Get a position grid that is not structured layer-by-layer but is simply a 2D array of shape (N_t*N_o, 3) where
        N_t is the length of translation grid and N_o the length of orientation grid.
        """
        return self._t_and_o_2_positions(o_property=self.get_o_grid().get_grid_as_array(only_upper=False),
                                         t_property=self.get_t_grid().get_trans_grid())

    def get_all_position_volumes(self) -> NDArray:
        # o grid has the option to get size of areas -> need to be divided by 3 and multiplied with radius^3 to get
        # volumes in the first shell, later shells need previous shells subtracted
        radius_above = self.get_between_radii()
        radius_below = np.concatenate(([0, ], radius_above[:-1]))
        area = self.get_o_grid().get_cell_volumes()
        cumulative_volumes = (self._t_and_o_2_positions(o_property=area/3, t_property=radius_above**3) -
                              self._t_and_o_2_positions(o_property=area/3, t_property=radius_below**3))
        return cumulative_volumes

    def _get_N_N_position_array(self, property="adjacency"):
        flat_pos_grid = self.get_position_grid_as_array()
        n_points = len(flat_pos_grid)  # equals n_o*n_t
        n_o = self.o_rotations.get_N()
        n_t = self.t_grid.get_N_trans()

        # First you have neighbours that occur from being at subsequent radii and the same ray
        # Since the position grid has all orientations at first r, then all at second r ... the points i and i+n_o will
        # always be neighbours, so we need the off-diagonals by n_o and -n_o
        # Most points have two neighbours this way, first and last layer have only one

        if property == "adjacency":
            my_diags = (True,)
            my_dtype = bool
            # within a layer
            neig = self.o_rotations.get_voronoi_adjacency(only_upper=False, include_opposing_neighbours=False).toarray()
            multiply = np.ones(n_t)
        elif property == "border":
            my_diags  = []
            radius_1_areas = self.o_rotations.get_cell_volumes()
            # last layer doesn't have a border up
            for layer_i, radius in enumerate(self.get_between_radii()[:-1]):
                my_diags.extend(radius_1_areas * radius**2)  # todo: test
            my_dtype = float
            # within a layer -> this is the arch above
            neig = self.o_rotations.get_center_distances(only_upper=False, include_opposing_neighbours=False).toarray()
            multiply = self.get_between_radii() ** 2 / 2
            print(multiply)
        elif property == "distance":
            # n_o elements will have the same distance
            increments = self.t_grid.get_increments()
            print(increments)
            my_diags = self._t_and_o_2_positions(np.ones(len(self.o_rotations)), increments) # todo: test
            my_dtype = float
        else:
            raise ValueError(f"Not recognised argument property={property}")

        same_ray_neighbours = diags(my_diags, offsets=n_o, shape=(n_points, n_points), dtype=my_dtype,
                                          format="coo")
        same_ray_neighbours += diags(my_diags, offsets=-n_o, shape=(n_points, n_points), dtype=my_dtype,
                                          format="coo")

        # Now we also want neighbours on the same level based on Voronoi discretisation
        # We first focus on the first n_o points since the set-up repeats at every radius

        # can't create Voronoi grid with <= 4 points, but then they are just all neighbours (except with itself)
        # if n_o <= 4:
        #     neig = np.ones((n_o, n_o), dtype=my_dtype) ^ np.eye(n_o, dtype=my_dtype)
        # else:
        # neig = self.o_rotations.get_voronoi_adjacency(only_upper=False, include_opposing_neighbours=False).toarray()

        # in case there are several translation distances, the array neig repeats along the diagonal n_t times
        if n_t > 1:
            my_blocks = [neig]
            my_blocks.extend([None,] * n_t)
            my_blocks = my_blocks * n_t
            my_blocks = my_blocks[:-n_t]
            my_blocks = np.array(my_blocks, dtype=object)
            my_blocks = my_blocks.reshape(n_t, n_t)
            same_radius_neighbours = bmat(my_blocks, dtype=float)

            #print(same_radius_neighbours.row, same_radius_neighbours.data)

            for ind_n_t in range(n_t):
                smallest_row = ind_n_t*n_o <= same_radius_neighbours.row
                largest_row = same_radius_neighbours.row < (ind_n_t+1)*n_o
                smallest_column = ind_n_t * n_o <= same_radius_neighbours.col
                largest_column = same_radius_neighbours.col < (ind_n_t + 1) * n_o
                mask = smallest_row & largest_row & smallest_column & largest_column
                same_radius_neighbours.data[mask] *= multiply[ind_n_t]
        else:
            same_radius_neighbours = coo_array(neig) * multiply
        all_neighbours = same_ray_neighbours + same_radius_neighbours
        return all_neighbours

    def get_adjacency_of_position_grid(self) -> coo_array:
        """
        Get a position grid adjacency matrix of shape (n_t*n_o, n_t*n_o) based on the numbering of flat position matrix.
        Two indices of position matrix are neigbours if
            i) one is directly above the other, or
            ii) they are voronoi neighbours at the same radius

        Returns:
            a diagonally-symmetric boolean sparse matrix where entries are True if neighbours and False otherwise

        """
        return self._get_N_N_position_array(property="adjacency")

    def get_borders_of_position_grid(self) -> coo_array:
        return self._get_N_N_position_array(property="border")

    def _change_voronoi_radius(self, sv: SphericalVoronoi, new_radius: float) -> SphericalVoronoi:
        """
        This is a helper function. Since a FullGrid consists of several layers of spheres in which the points are at
        exactly same places (just at different radii), it makes sense not to recalculate, but just to scale the radius,
        vertices and points out of which the SphericalVoronoi consists to a new radius.
        """
        sv.radius = new_radius
        sv.vertices = normalise_vectors(sv.vertices, length=new_radius)
        sv.points = normalise_vectors(sv.points, length=new_radius)
        # important that it's a copy!
        return copy(sv)

    def get_position_voronoi(self):
        unit_sph_voronoi = self.o_rotations.get_spherical_voronoi_cells()
        between_radii = self.get_between_radii()
        all_sv = [self._change_voronoi_radius(unit_sph_voronoi, r) for r in between_radii]
        return all_sv


#
#     ###################################################################################################################
#     #                           useful properties
#     ###################################################################################################################
#
#     def find_voronoi_vertices_of_point(self, point_index: int, which: str = "all") -> NDArray:
#         """
#         Using an index (from flattened position grid), find which voronoi vertices belong to this point.
#
#         Args:
#             point_index: for which point in flattened position grid the vertices should be found
#             which: which vertices: all, upper or lower
#
#         Returns:
#             an array of vertices, each row is a 3D point.
#         """
#         my_point = Point(point_index, self)
#
#         if which == "all":
#             vertices = my_point.get_vertices()
#         elif which == "upper":
#             vertices = my_point.get_vertices_above()
#         elif which == "lower":
#             vertices = my_point.get_vertices_below()
#         else:
#             raise ValueError("The value of which not recognised, select 'all', 'upper', 'lower'.")
#         return vertices
#
#     def get_distance_between_centers(self, index_1: int, index_2: int, print_message=True) -> Optional[float]:
#         """
#         Calculate the distance between two position grid points selected by their indices. Optionally print message
#         if they are not neighbours.
#
#         There are three options:
#             - point1 is right above point2 or vide versa -> the distance is measured in a straight line from the center
#             - point1 and point2 are sideways neighbours -> the distance is measured on the circumference of their radius
#             - point1 and point2 are not neighbours -> return None
#
#         Returns:
#             None if not neighbours, distance in angstrom else
#         """
#         point_1 = Point(index_1, self)
#         point_2 = Point(index_2, self)
#
#         if self._point1_right_above_point2(point_1, point_2) or self._point2_right_above_point1(point_1, point_2):
#             return np.abs(point_1.d_to_origin - point_2.d_to_origin)
#         elif self._are_sideways_neighbours(point_1, point_2):
#             radius = point_1.d_to_origin
#             theta = angle_between_vectors(point_1.point, point_2.point)
#             # length of arc
#             return theta * radius
#         else:
#             if print_message:
#                 print(f"Points {index_1} and {index_2} are not neighbours.")
#             return None
#
#     def get_division_area(self, index_1: int, index_2: int, print_message: bool = True) -> Optional[float]:
#         """
#         Calculate the area (in Angstrom squared) that is the border area between two Voronoi cells. This is either
#         a curved area (a part of a sphere) if the two cells are one above the other or a flat, part of circle or
#         circular ring if the cells are neighbours at the same level. If points are not neighbours, returns None.
#         """
#         point_1 = Point(index_1, self)
#         point_2 = Point(index_2, self)
#
#         # if they are sideways neighbours
#         if self._at_same_radius(point_1, point_2):
#             # vertices_above
#             vertices_1a = point_1.get_vertices_above()
#             vertices_2a = point_2.get_vertices_above()
#             r_larger = np.linalg.norm(vertices_1a[0])
#             set_vertices_1a = set([tuple(v) for v in vertices_1a])
#             set_vertices_2a = set([tuple(v) for v in vertices_2a])
#             # vertices that are above point 1 and point 2
#             intersection_a = set_vertices_1a.intersection(set_vertices_2a)
#             # vertices below - only important to determine radius
#             vertices_1b = point_1.get_vertices_below()
#             r_smaller = np.linalg.norm(vertices_1b[0])
#             if len(intersection_a) != 2:
#                 if print_message:
#                     print(f"Points {index_1} and {index_2} are not neighbours.")
#                 return None
#             else:
#                 # angle will be determined by the vector from origin to both points above
#                 intersection_list = list(intersection_a)
#                 theta = angle_between_vectors(np.array(intersection_list[0]), np.array(intersection_list[1]))
#                 return theta / 2 * (r_larger ** 2 - r_smaller ** 2)
#         # if point_1 right above point_2
#         if self._point1_right_above_point2(point_1, point_2):
#             return point_2.get_area_above()
#         if self._point2_right_above_point1(point_1, point_2):
#             return point_1.get_area_above()
#         # if no exit point so far
#         if print_message:
#             print(f"Points {index_1} and {index_2} are not neighbours.")
#         return None
#
#     def get_volume(self, index: int) -> float:
#         """
#         Get the volume of any cell in FullVoronoiGrid, defined by its index in flattened position grid.
#         """
#         point = Point(index, self)
#         return point.get_cell_volume()
#
#     @save_or_use_saved
#     def get_all_voronoi_volumes(self) -> NDArray:
#         """
#         Get an array in the same order as flat position grid, listing the volumes of Voronoi cells.
#         """
#         N = len(self.flat_positions)
#         volumes = np.zeros((N,))
#         for i in range(0, N):
#             volumes[i] = self.get_volume(i)
#         return volumes
#
#     def _get_property_all_pairs(self, method: Callable) -> csc_array:
#         """
#         Helper method for any property that is dependent on a pair of indices. Examples: obtaining all distances
#         between cell centers or all areas between cells. Always symmetrical - value at (i, j) equals the one at (j, i)
#
#         Args:
#             method: must have a signature (index1: int, index2: int, print_message: boolean)
#
#         Returns:
#             a sparse matrix of shape (len_flat_position_array, len_flat_position_array)
#         """
#         data = []
#         row_indices = []
#         column_indices = []
#         N_pos_array = len(self.flat_positions)
#         for i in range(N_pos_array):
#             for j in range(i + 1, N_pos_array):
#                 my_property = method(i, j, print_message=False)
#                 if my_property is not None:
#                     # this value will be added to coordinates (i, j) and (j, i)
#                     data.extend([my_property, my_property])
#                     row_indices.extend([i, j])
#                     column_indices.extend([j, i])
#         sparse_property = coo_array((data, (row_indices, column_indices)), shape=(N_pos_array, N_pos_array))
#         return sparse_property.tocsc()
#
#     @save_or_use_saved
#     def get_all_voronoi_surfaces(self) -> csc_array:
#         """
#         If l is the length of the flattened position array, returns a lxl (sparse) array where the i-th row and j-th
#         column (as well as the j-th row and the i-th column) represent the size of the Voronoi surface between points
#         i and j in position grid. If the points do not share a division area, no value will be set.
#         """
#         return self._get_property_all_pairs(self.get_division_area)
#
#     @save_or_use_saved
#     def get_all_distances_between_centers(self) -> csc_array:
#         """
#         Get a sparse matrix where for all sets of neighbouring cells the distance between Voronoi centers is provided.
#         Therefore, the value at [i][j] equals the value at [j][i]. For more information, check
#         self.get_distance_between_centers.
#         """
#         return self._get_property_all_pairs(self.get_distance_between_centers)
#
#     def get_all_voronoi_surfaces_as_numpy(self) -> NDArray:
#         """See self.get_all_voronoi_surfaces, only transforms sparse array to normal array."""
#         return self.get_all_voronoi_surfaces().toarray(order='C')
#
#     def get_all_distances_between_centers_as_numpy(self) -> NDArray:
#         """See self.get_all_distances_between_centers, only transforms sparse array to normal array."""
#         return self.get_all_distances_between_centers().toarray(order='C')
#
#
# class Point:
#
#     """
#     A Point represents a single cell in a particular FullVoronoiGrid. It holds all relevant information, eg. the
#     index of the cell within a single layer and in a full flattened position grid. It enables the identification
#     of Voronoi vertices and connected calculations (distances, areas, volumes).
#     """
#
#     def __init__(self, index_position_grid: int, full_sv: FullVoronoiGrid):
#         self.full_sv = full_sv
#         self.index_position_grid: int = index_position_grid
#         self.point: NDArray = self.full_sv.flat_positions[index_position_grid]
#         self.d_to_origin: float = np.linalg.norm(self.point)
#         self.index_radial: int = self._find_index_radial()
#         self.index_within_sphere: int = self._find_index_within_sphere()
#
#     def get_normalised_point(self) -> NDArray:
#         """Get the vector to the grid point (center of Voronoi cell) normalised to length 1."""
#         return normalise_vectors(self.point, length=1)
#
#     def _find_index_radial(self) -> int:
#         """Find to which radial layer of points this point belongs."""
#         point_radii = self.full_sv.full_grid.get_radii()
#         for i, dist in enumerate(point_radii):
#             if np.isclose(dist, self.d_to_origin):
#                 return i
#         else:
#             raise ValueError("The norm of the point not close to any of the radii.")
#
#     def _find_index_within_sphere(self) -> int:
#         """Find the index of the point within a single layer of possible orientations."""
#         radial_index = self.index_radial
#         num_o_rot = len(self.full_sv.full_grid.o_rotations.get_grid_as_array())
#         return self.index_position_grid - num_o_rot * radial_index
#
#     def _find_index_sv_above(self) -> Optional[int]:
#         for i, sv in enumerate(self.full_sv.get_voronoi_discretisation()):
#             if sv.radius > self.d_to_origin:
#                 return i
#         else:
#             # the point is outside the largest voronoi sphere
#             return None
#
#     def _get_sv_above(self) -> SphericalVoronoi:
#         """Get the spherical Voronoi with the first radius that is larger than point radius."""
#         return self.full_sv.get_voronoi_discretisation()[self._find_index_sv_above()]
#
#     def _get_sv_below(self) -> Optional[SphericalVoronoi]:
#         """Get the spherical Voronoi with the largest radius that is smaller than point radius. If the point is in the
#         first layer, return None."""
#         index_above = self._find_index_sv_above()
#         if index_above != 0:
#             return self.full_sv.get_voronoi_discretisation()[index_above-1]
#         else:
#             return None
#
#     ##################################################################################################################
#     #                            GETTERS - DISTANCES, AREAS, VOLUMES
#     ##################################################################################################################
#
#     def get_radius_above(self) -> float:
#         """Get the radius of the SphericalVoronoi cell that is the upper surface of the cell."""
#         sv_above = self._get_sv_above()
#         return sv_above.radius
#
#     def get_radius_below(self) -> float:
#         """Get the radius of the SphericalVoronoi cell that is the lower surface of the cell (return zero if there is
#         no Voronoi layer below)."""
#         sv_below = self._get_sv_below()
#         if sv_below is None:
#             return 0.0
#         else:
#             return sv_below.radius
#
#     def get_area_above(self) -> float:
#         """Get the area of the Voronoi surface that is the upper side of the cell (curved surface, part of sphere)."""
#         sv_above = self._get_sv_above()
#         areas = sv_above.calculate_areas()
#         return areas[self.index_within_sphere]
#
#     def get_area_below(self) -> float:
#         """Get the area of the Voronoi surface that is the lower side of the cell (curved surface, part of sphere)."""
#         sv_below = self._get_sv_below()
#         if sv_below is None:
#             return 0.0
#         else:
#             areas = sv_below.calculate_areas()
#             return areas[self.index_within_sphere]
#
#     def get_vertices_above(self) -> NDArray:
#         """Get the vertices of this cell that belong to the SphericalVoronoi above the point."""
#         sv_above = self._get_sv_above()
#         regions = sv_above.regions[self.index_within_sphere]
#         vertices_above = sv_above.vertices[regions]
#         return vertices_above
#
#     def get_vertices_below(self) -> NDArray:
#         """Get the vertices of this cell that belong to the SphericalVoronoi below the point (or just the origin
#         if the point belongs to the first layer."""
#         sv_below = self._get_sv_below()
#         if sv_below is None:
#             vertices_below = np.zeros((1, 3))
#         else:
#             regions = sv_below.regions[self.index_within_sphere]
#             vertices_below = sv_below.vertices[regions]
#
#         return vertices_below
#
#     def get_vertices(self) -> NDArray:
#         """Get all vertices of this cell as a single array."""
#         vertices_above = self.get_vertices_above()
#         vertices_below = self.get_vertices_below()
#
#         return np.concatenate((vertices_above, vertices_below))
#
#     def get_cell_volume(self) -> float:
#         """Get the volume of this cell (calculated as the difference between the part of sphere volume at radius above
#         minus the same area at smaller radius)."""
#         radius_above = self.get_radius_above()
#         radius_below = self.get_radius_below()
#         area_above = self.get_area_above()
#         area_below = self.get_area_below()
#         volume = 1/3 * (radius_above * area_above - radius_below * area_below)
#         return volume


class ConvergenceFullGridO:
    
    """
    This object is used to study the convergence of properties as the number of points in the o_grid of FullGrid 
    changes. Mostly used to plot how voronoi volumes, calculation times ... converge.
    """

    def __init__(self, b_grid_name: str, t_grid_name: str,  o_alg_name: str, N_set: tuple = None,
                 use_saved: bool = False, **kwargs):
        if N_set is None:
            N_set = SMALL_NS
        self.N_set = N_set
        self.alg_name = o_alg_name
        self.use_saved = use_saved
        self.list_full_grids = self.create(b_grid_name=b_grid_name, t_grid_name=t_grid_name,  o_alg_name=o_alg_name,
                                           N_set=self.N_set, use_saved=use_saved, **kwargs)

    def get_name(self) -> str:
        """Get name for saving."""
        b_name = self.list_full_grids[0].b_rotations.get_name(with_dim=False)
        t_name = self.list_full_grids[0].t_grid.grid_hash
        return f"convergence_o_{self.alg_name}_b_{b_name}_t_{t_name}"

    @classmethod
    def create(cls,  b_grid_name: str, t_grid_name: str,  o_alg_name: str, N_set: tuple, **kwargs) -> list:
        """
        Build the FullGrid object for each of the given Ns in N_set. This is the create function that is only
        called by __init__.
        """
        list_full_grids = []
        for N in N_set:
            fg = FullGrid(b_grid_name=b_grid_name, o_grid_name=f"{o_alg_name}_{N}", t_grid_name=t_grid_name, **kwargs)
            list_full_grids.append(fg)
        return list_full_grids

    @save_or_use_saved
    def get_voronoi_volumes(self) -> pd.DataFrame:
        """
        Get all volumes of cells for all choices of self.N_set in order to create convergence plots.
        """
        data = []
        for N, fg in zip(self.N_set, self.list_full_grids):
            N_real = fg.o_rotations.get_N()
            vor_radius = list(fg.get_between_radii())
            vor_radius.insert(0, 0)
            vor_radius = np.array(vor_radius)
            fvg = fg.get_full_voronoi_grid()
            ideal_volumes = 4/3 * pi * (vor_radius[1:]**3 - vor_radius[:-1]**3) / N_real
            real_volumes = fvg.get_all_voronoi_volumes()
            for i, volume in enumerate(real_volumes):
                layer = i//N_real
                data.append([N_real, layer, ideal_volumes[i//N_real], volume])
        df = pd.DataFrame(data, columns=["N", "layer", "ideal volume", "Voronoi cell volume"])
        return df

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from molgri.plotting.spheregrid_plots import PolytopePlot, SphereGridPlot
    from molgri.space.polytopes import PolyhedronFromG
    import seaborn as sns

    n_o = 7
    n_b = 40
    fg = FullGrid(f"fulldiv_{n_b}", f"ico_{n_o}", "linspace(1, 5, 4)", use_saved=False)

    print(fg.get_name())
    print(fg.get_position_grid())
    print(fg.get_adjacency_of_orientation_grid())

    # position_adjacency = fg.get_adjacency_of_position_grid().toarray()
    # orientation_adjacency = fg.get_adjacency_of_orientation_grid().toarray()
    # full_adjacency = fg.get_full_adjacency().toarray()
    #
    # fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    # sns.heatmap(position_adjacency, cmap="gray", ax=ax[0])
    # sns.heatmap(orientation_adjacency, cmap="gray", ax=ax[1])
    # sns.heatmap(full_adjacency, cmap="gray", ax=ax[2])
    # ax[0].set_title("Position adjacency")
    # ax[1].set_title("Orientation adjacency")
    # ax[2].set_title("Full adjacency")
    # plt.tight_layout()
    # plt.show()

