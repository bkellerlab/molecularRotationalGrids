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
"""
from __future__ import annotations
from copy import copy
from typing import Callable, Tuple, Optional


import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import cdist
from scipy.constants import pi
from scipy.sparse import coo_array, csc_array
import pandas as pd

from molgri.constants import SMALL_NS
from molgri.space.rotobj import SphereGridFactory, SphereGridNDim
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

    def __init__(self, b_grid_name: str, o_grid_name: str, t_grid_name: str, use_saved: bool = True,
                 filter_non_unique: bool = False):

        """
        Args:
            b_grid_name: of the form 'ico_17'
            o_grid_name: of the form 'cube4D_12'
            t_grid_name: of the form '[1, 3, 4.5]'
            use_saved: try to obtain saved data if possible
            filter_non_unique: remove repeating points from spherical grids so that Voronoi cells can be built
        """
        b_grid_name = GridNameParser(b_grid_name, "b")
        self.b_rotations = SphereGridFactory.create(alg_name=b_grid_name.get_alg(), N=b_grid_name.get_N(),
                                                    dimensions=4, use_saved=use_saved,
                                                    filter_non_unique=filter_non_unique)
        o_grid_name = GridNameParser(o_grid_name, "o")
        self.o_rotations = SphereGridFactory.create(alg_name=o_grid_name.get_alg(), N=o_grid_name.get_N(),
                                                    dimensions=3, use_saved=use_saved,
                                                    filter_non_unique=filter_non_unique)
        self.o_positions = self.o_rotations.get_grid_as_array()
        self.t_grid = TranslationParser(t_grid_name)
        self.use_saved = use_saved

    def get_name(self) -> str:
        """Name that is appropriate for saving."""
        o_name = self.o_rotations.get_name(with_dim=False)
        b_name = self.b_rotations.get_name(with_dim=False)
        return f"o_{o_name}_b_{b_name}_t_{self.t_grid.grid_hash}"

    @save_or_use_saved
    def get_full_grid(self) -> Tuple[SphereGridNDim, SphereGridNDim, TranslationParser]:
        """Get all relevant sub-grids."""
        return self.b_rotations, self.o_rotations, self.t_grid

    def get_radii(self) -> NDArray:
        """
        Get the radii at which points are positioned. Result is in Angstroms.
        """
        return self.t_grid.get_trans_grid()

    def get_between_radii(self) -> NDArray:
        """
        Get the radii at which Voronoi cells of the position grid should be positioned. This should be right in-between
        two orientation point layers (except the first layer that is fully encapsulated by the first voronoi layer
        and the last one that is simply in-between two voronoi layers).

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

    def get_body_rotations(self) -> Rotation:
        """Get a Rotation object (may encapsulate a list of rotations) from the body grid."""
        return Rotation.from_quat(self.b_rotations.get_grid_as_array())

    @save_or_use_saved
    def get_position_grid(self) -> NDArray:
        """
        Get a 'product' of o_grid and t_grid so you can visualise points in space at which COM of the second molecule
        will be positioned. Important: Those are points on spheres in real space.

        Returns:
            an array of shape (len_o_grid, len_t_grid, 3) in which the elements of result[0] have the first
            rotational position, each line at a new (increasing) distance, result[1] the next rotational position,
            again at all possible distances ...
        """
        dist_array = self.get_radii()
        o_grid = self.o_positions
        num_dist = len(dist_array)
        num_orient = len(o_grid)
        result = np.zeros((num_dist, num_orient, 3))
        for i, dist in enumerate(dist_array):
            result[i] = np.multiply(o_grid, dist)
            norms = norm_per_axis(result[i])
            assert np.allclose(norms, dist), "In a position grid, all vectors in i-th 'row' should have the same norm!"
        result = np.swapaxes(result, 0, 1)
        return result

    @save_or_use_saved
    def get_flat_position_grid(self) -> NDArray:
        """
        Get a position grid that is not structured layer-by-layer but is simply a 2D array of shape (N_t*N_o, 3) where
        N_t is the length of translation grid and N_o the length of orientation grid.
        """
        pos_grid = self.get_position_grid()
        pos_grid = np.swapaxes(pos_grid, 0, 1)
        return pos_grid.reshape((-1, 3))

    @deprecated
    def save_full_grid(self):
        """Save position grid."""
        np.save(f"{PATH_OUTPUT_FULL_GRIDS}position_grid_{self.get_name()}", self.get_position_grid())

    def get_full_voronoi_grid(self) -> Optional[FullVoronoiGrid]:
        """Get the corresponding FullVoronoiGrid object."""
        try:
            return FullVoronoiGrid(self)
        except AttributeError:
            return None

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

    def points2cell_scipy(self, points_vector: NDArray) -> NDArray:
        """
        For comparison to self.point2cell_position_grid(), here is a method that assigns belonging to a grid point
        simply based on distance and not considering the spherical nature of the grid.
        """
        my_cdist = cdist(points_vector, self.get_flat_position_grid())
        return np.argmin(my_cdist, axis=1)


class FullVoronoiGrid:

    """
    Created from FullGrid, implements functionality that is specifically Voronoi-based.

    Enables calculations of all distances, areas, and volumes of/between cells.
    """

    def __init__(self, full_grid: FullGrid):
        self.full_grid = full_grid
        self.use_saved = self.full_grid.use_saved
        self.flat_positions = self.full_grid.get_flat_position_grid()
        self.all_sv = None
        self.get_voronoi_discretisation()

    ###################################################################################################################
    #                           basic creation/getter functions
    ###################################################################################################################

    def get_name(self) -> str:
        """Name for saving files."""
        return f"voronoi_{self.full_grid.get_name()}"

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

    @save_or_use_saved
    def get_voronoi_discretisation(self) -> list:
        """
        Create a list of spherical voronoi-s that are identical except at different radii. The radii are set in such a
        way that there is always a Voronoi cell layer right in-between two layers of grid cells. (see FullGrid method
        get_between_radii for details.
        """
        if self.all_sv is None:
            unit_sph_voronoi = self.full_grid.o_rotations.get_spherical_voronoi_cells()
            between_radii = self.full_grid.get_between_radii()
            self.all_sv = [self._change_voronoi_radius(unit_sph_voronoi, r) for r in between_radii]
        return self.all_sv

    ###################################################################################################################
    #                           point helper functions
    ###################################################################################################################

    def _at_same_radius(self, point1: Point, point2: Point) -> bool:
        """Check that two points belong to the same layer"""
        return np.isclose(point1.d_to_origin, point2.d_to_origin)

    def _are_neighbours(self, point1: Point, point2: Point) -> bool:
        """
        Check whether points are neighbours in any way. We define neighbours as those sharing surface and NOT sharing
        only vertex points or lines.
        """
        index1 = point1.index_position_grid
        index2 = point2.index_position_grid
        return self.get_division_area(index1, index2, print_message=False) is not None

    def _are_sideways_neighbours(self, point1: Point, point2: Point) -> bool:
        """
        Check whether points belong to the same radius and are neighbours.
        """
        return self._at_same_radius(point1, point2) and self._are_neighbours(point1, point2)

    def _are_on_same_ray(self, point1: Point, point2: Point) -> bool:
        """
        Two points are on the same ray if they are on the same vector from origin (may be at different distances)
        """
        normalised1 = point1.get_normalised_point()
        normalised2 = point2.get_normalised_point()
        return np.allclose(normalised1, normalised2)

    def _point1_right_above_point2(self, point1: Point, point2: Point) -> bool:
        """
        Right above should be interpreted spherically.

        Conditions: on the same ray + radius of point1 one unit bigger
        """
        radial_index1 = point1.index_radial
        radial_index2 = point2.index_radial
        return self._are_on_same_ray(point1, point2) and radial_index1 == radial_index2 + 1

    def _point2_right_above_point1(self, point1: Point, point2: Point) -> bool:
        """See _point1_right_above_point2."""
        radial_index1 = point1.index_radial
        radial_index2 = point2.index_radial
        return self._are_on_same_ray(point1, point2) and radial_index1 + 1 == radial_index2

    ###################################################################################################################
    #                           useful properties
    ###################################################################################################################

    def find_voronoi_vertices_of_point(self, point_index: int, which: str = "all") -> NDArray:
        """
        Using an index (from flattened position grid), find which voronoi vertices belong to this point.

        Args:
            point_index: for which point in flattened position grid the vertices should be found
            which: which vertices: all, upper or lower

        Returns:
            an array of vertices, each row is a 3D point.
        """
        my_point = Point(point_index, self)

        if which == "all":
            vertices = my_point.get_vertices()
        elif which == "upper":
            vertices = my_point.get_vertices_above()
        elif which == "lower":
            vertices = my_point.get_vertices_below()
        else:
            raise ValueError("The value of which not recognised, select 'all', 'upper', 'lower'.")
        return vertices

    def get_distance_between_centers(self, index_1: int, index_2: int, print_message=True) -> Optional[float]:
        """
        Calculate the distance between two position grid points selected by their indices. Optionally print message
        if they are not neighbours.

        There are three options:
            - point1 is right above point2 or vide versa -> the distance is measured in a straight line from the center
            - point1 and point2 are sideways neighbours -> the distance is measured on the circumference of their radius
            - point1 and point2 are not neighbours -> return None

        Returns:
            None if not neighbours, distance in angstrom else
        """
        point_1 = Point(index_1, self)
        point_2 = Point(index_2, self)

        if self._point1_right_above_point2(point_1, point_2) or self._point2_right_above_point1(point_1, point_2):
            return np.abs(point_1.d_to_origin - point_2.d_to_origin)
        elif self._are_sideways_neighbours(point_1, point_2):
            radius = point_1.d_to_origin
            theta = angle_between_vectors(point_1.point, point_2.point)
            # length of arc
            return theta * radius
        else:
            if print_message:
                print(f"Points {index_1} and {index_2} are not neighbours.")
            return None

    def get_division_area(self, index_1: int, index_2: int, print_message: bool = True) -> Optional[float]:
        """
        Calculate the area (in Angstrom squared) that is the border area between two Voronoi cells. This is either
        a curved area (a part of a sphere) if the two cells are one above the other or a flat, part of circle or
        circular ring if the cells are neighbours at the same level. If points are not neighbours, returns None.
        """
        point_1 = Point(index_1, self)
        point_2 = Point(index_2, self)

        # if they are sideways neighbours
        if self._at_same_radius(point_1, point_2):
            # vertices_above
            vertices_1a = point_1.get_vertices_above()
            vertices_2a = point_2.get_vertices_above()
            r_larger = np.linalg.norm(vertices_1a[0])
            set_vertices_1a = set([tuple(v) for v in vertices_1a])
            set_vertices_2a = set([tuple(v) for v in vertices_2a])
            # vertices that are above point 1 and point 2
            intersection_a = set_vertices_1a.intersection(set_vertices_2a)
            # vertices below - only important to determine radius
            vertices_1b = point_1.get_vertices_below()
            r_smaller = np.linalg.norm(vertices_1b[0])
            if len(intersection_a) != 2:
                if print_message:
                    print(f"Points {index_1} and {index_2} are not neighbours.")
                return None
            else:
                # angle will be determined by the vector from origin to both points above
                intersection_list = list(intersection_a)
                theta = angle_between_vectors(np.array(intersection_list[0]), np.array(intersection_list[1]))
                return theta / 2 * (r_larger ** 2 - r_smaller ** 2)
        # if point_1 right above point_2
        if self._point1_right_above_point2(point_1, point_2):
            return point_2.get_area_above()
        if self._point2_right_above_point1(point_1, point_2):
            return point_1.get_area_above()
        # if no exit point so far
        if print_message:
            print(f"Points {index_1} and {index_2} are not neighbours.")
        return None

    def get_volume(self, index: int) -> float:
        """
        Get the volume of any
        """
        point = Point(index, self)
        return point.get_cell_volume()

    @save_or_use_saved
    def get_all_voronoi_volumes(self) -> NDArray:
        """
        Get an array in the same order as flat position grid, listing the volumes of Voronoi cells.
        """
        N = len(self.flat_positions)
        volumes = np.zeros((N,))
        for i in range(0, N):
            volumes[i] = self.get_volume(i)
        return volumes

    def _get_property_all_pairs(self, method: Callable) -> csc_array:
        """
        Helper method for any property that is dependent on a pair of indices. Examples: obtaining all distances
        between cell centers or all areas between cells. Always symmetrical - value at (i, j) equals the one at (j, i)

        Args:
            method: must have a signature (index1: int, index2: int, print_message: boolean)

        Returns:
            a sparse matrix of shape (len_flat_position_array, len_flat_position_array)
        """
        data = []
        row_indices = []
        column_indices = []
        N_pos_array = len(self.flat_positions)
        for i in range(N_pos_array):
            for j in range(i + 1, N_pos_array):
                my_property = method(i, j, print_message=False)
                if my_property is not None:
                    # this value will be added to coordinates (i, j) and (j, i)
                    data.extend([my_property, my_property])
                    row_indices.extend([i, j])
                    column_indices.extend([j, i])
        sparse_property = coo_array((data, (row_indices, column_indices)), shape=(N_pos_array, N_pos_array))
        return sparse_property.tocsc()

    @save_or_use_saved
    def get_all_voronoi_surfaces(self) -> csc_array:
        """
        If l is the length of the flattened position array, returns a lxl (sparse) array where the i-th row and j-th
        column (as well as the j-th row and the i-th column) represent the size of the Voronoi surface between points
        i and j in position grid. If the points do not share a division area, no value will be set.
        """
        return self._get_property_all_pairs(self.get_division_area)

    @save_or_use_saved
    def get_all_distances_between_centers(self) -> csc_array:
        """
        Get a sparse matrix where for all sets of neighbouring cells the distance between Voronoi centers is provided.
        Therefore, the value at [i][j] equals the value at [j][i]. For more information, check
        self.get_distance_between_centers.
        """
        return self._get_property_all_pairs(self.get_distance_between_centers)

    def get_all_voronoi_surfaces_as_numpy(self) -> NDArray:
        """See self.get_all_voronoi_surfaces, only transforms sparse array to normal array."""
        return self.get_all_voronoi_surfaces().toarray(order='C')

    def get_all_distances_between_centers_as_numpy(self) -> NDArray:
        """See self.get_all_distances_between_centers, only transforms sparse array to normal array."""
        return self.get_all_distances_between_centers().toarray(order='C')


class Point:

    """
    A Point represents a single cell in a particular FullVoronoiGrid. It holds all relevant information, eg. the
    index of the cell within a single layer and in a full flattened position grid. It enables the identification
    of Voronoi vertices and connected calculations (distances, areas, volumes).
    """

    def __init__(self, index_position_grid: int, full_sv: FullVoronoiGrid):
        self.full_sv = full_sv
        self.index_position_grid = index_position_grid
        self.point = self.full_sv.flat_positions[index_position_grid]
        self.d_to_origin = np.linalg.norm(self.point)
        self.index_radial = self._find_index_radial()
        self.index_within_sphere = self._find_index_within_sphere()

    def get_normalised_point(self) -> NDArray:
        """Get the vector to the grid point (center of Voronoi cell) normalised to length 1."""
        return normalise_vectors(self.point, length=1)

    def _find_index_radial(self) -> int:
        """Find to which radial layer of points this point belongs."""
        point_radii = self.full_sv.full_grid.get_radii()
        for i, dist in enumerate(point_radii):
            if np.isclose(dist, self.d_to_origin):
                return i
        else:
            raise ValueError("The norm of the point not close to any of the radii.")

    def _find_index_within_sphere(self) -> int:
        """Find the index of the point within a single layer of possible orientations."""
        radial_index = self.index_radial
        num_o_rot = len(self.full_sv.full_grid.o_rotations.get_grid_as_array())
        return self.index_position_grid - num_o_rot * radial_index

    def _find_index_sv_above(self) -> Optional[int]:
        for i, sv in enumerate(self.full_sv.get_voronoi_discretisation()):
            if sv.radius > self.d_to_origin:
                return i
        else:
            # the point is outside the largest voronoi sphere
            return None

    def _get_sv_above(self) -> SphericalVoronoi:
        """Get the spherical Voronoi with the first radius that is larger than point radius."""
        return self.full_sv.get_voronoi_discretisation()[self._find_index_sv_above()]

    def _get_sv_below(self) -> Optional[SphericalVoronoi]:
        """Get the spherical Voronoi with the largest radius that is smaller than point radius. If the point is in the
        first layer, return None."""
        index_above = self._find_index_sv_above()
        if index_above != 0:
            return self.full_sv.get_voronoi_discretisation()[index_above-1]
        else:
            return None

    ##################################################################################################################
    #                            GETTERS - DISTANCES, AREAS, VOLUMES
    ##################################################################################################################

    def get_radius_above(self) -> float:
        """Get the radius of the SphericalVoronoi cell that is the upper surface of the cell."""
        sv_above = self._get_sv_above()
        return sv_above.radius

    def get_radius_below(self) -> float:
        """Get the radius of the SphericalVoronoi cell that is the lower surface of the cell (return zero if there is
        no Voronoi layer below)."""
        sv_below = self._get_sv_below()
        if sv_below is None:
            return 0.0
        else:
            return sv_below.radius

    def get_area_above(self) -> float:
        """Get the area of the Voronoi surface that is the upper side of the cell (curved surface, part of sphere)."""
        sv_above = self._get_sv_above()
        areas = sv_above.calculate_areas()
        return areas[self.index_within_sphere]

    def get_area_below(self) -> float:
        """Get the area of the Voronoi surface that is the lower side of the cell (curved surface, part of sphere)."""
        sv_below = self._get_sv_below()
        if sv_below is None:
            return 0.0
        else:
            areas = sv_below.calculate_areas()
            return areas[self.index_within_sphere]

    def get_vertices_above(self) -> NDArray:
        """Get the vertices of this cell that belong to the SphericalVoronoi above the point."""
        sv_above = self._get_sv_above()
        regions = sv_above.regions[self.index_within_sphere]
        vertices_above = sv_above.vertices[regions]
        return vertices_above

    def get_vertices_below(self) -> NDArray:
        """Get the vertices of this cell that belong to the SphericalVoronoi below the point (or just the origin
        if the point belongs to the first layer."""
        sv_below = self._get_sv_below()
        if sv_below is None:
            vertices_below = np.zeros((1, 3))
        else:
            regions = sv_below.regions[self.index_within_sphere]
            vertices_below = sv_below.vertices[regions]

        return vertices_below

    def get_vertices(self):
        """Get all vertices of this cell as a single array."""
        vertices_above = self.get_vertices_above()
        vertices_below = self.get_vertices_below()

        return np.concatenate((vertices_above, vertices_below))

    def get_cell_volume(self):
        """Get the volume of this cell (calculated as the difference between the part of sphere volume at radius above
        minus the same area at smaller radius)."""
        radius_above = self.get_radius_above()
        radius_below = self.get_radius_below()
        area_above = self.get_area_above()
        area_below = self.get_area_below()
        volume = 1/3 * (radius_above * area_above - radius_below * area_below)
        return volume


class ConvergenceFullGridO:

    def __init__(self, b_grid_name: str, t_grid_name: str,  o_alg_name: str, N_set = None, use_saved=False, **kwargs):
        if N_set is None:
            N_set = SMALL_NS
        self.N_set = N_set
        self.alg_name = o_alg_name
        self.use_saved = use_saved
        self.list_full_grids = self.create(b_grid_name=b_grid_name, t_grid_name=t_grid_name,  o_alg_name=o_alg_name,
                                           N_set=self.N_set, use_saved=use_saved, **kwargs)

    def get_name(self):
        b_name = self.list_full_grids[0].b_rotations.get_name(with_dim=False)
        t_name = self.list_full_grids[0].t_grid.grid_hash
        return f"convergence_o_{self.alg_name}_b_{b_name}_t_{t_name}"

    @classmethod
    def create(cls,  b_grid_name: str, t_grid_name: str,  o_alg_name: str, N_set, **kwargs) -> list:
        list_full_grids = []
        for N in N_set:
            fg = FullGrid(b_grid_name=b_grid_name, o_grid_name=f"{o_alg_name}_{N}", t_grid_name=t_grid_name, **kwargs)
            list_full_grids.append(fg)
        return list_full_grids

    @save_or_use_saved
    def get_voronoi_volumes(self):
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
    full_grid = FullGrid(b_grid_name="cube3D_16", o_grid_name="ico_15", t_grid_name="[1, 2, 3]")
    full_grid.save_full_grid()
    # fvg = full_grid.get_full_voronoi_grid()
    # print(fvg)
    points = np.array([[-1, 3, 2], [-0.5, -0.5, 1], [22, 8, 4], [1, 1, 222]])
    print(full_grid.point2cell_position_grid(points))