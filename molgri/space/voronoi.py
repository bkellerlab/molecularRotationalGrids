"""
Building up on scipy objects Voronoi and SphericalVoronoi

"""
from abc import ABC, abstractmethod
from copy import copy
from itertools import combinations
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.spatial import SphericalVoronoi, ConvexHull
from scipy.spatial.distance import cdist

from molgri.constants import UNIQUE_TOL
from molgri.space.rotobj import SphereGrid3DFactory, SphereGrid4DFactory
from molgri.space.translations import get_between_radii
from molgri.space.utils import dist_on_sphere, distance_between_quaternions, find_inverse_quaternion, k_is_a_row, \
    normalise_vectors, \
    q_in_upper_sphere, \
    which_row_is_k


class AbstractVoronoi(ABC):

    """
    This is a general class I use as an extension of scipy Voronoi class. At the minimum, each subclass must
    implement getters for:
    1) centers
    2) vertices
    3) regions
    """

    def __init__(self):
        self.centers, self.vertices, self.regions = self._create_centers_vertices_regions()
        self.reduced_vertices, self.reduced_regions = self.get_reduced_vertices_regions()

    @abstractmethod
    def _create_centers_vertices_regions(self) -> Tuple:
        pass

    def get_all_voronoi_centers(self) -> NDArray:
        """
        Getter equivalent to scipy's sv.points.

        Returns:
            An array of points that define the cells of voronoi grid; the order is used for constructing indices of
            cells
        """
        return self.centers

    def get_all_voronoi_vertices(self, reduced=False) -> NDArray:
        """
        Getter equivalent to scipy's sv.vertices.

        Returns:
            An array of points that define the edges of voronoi grid; the order is used for constructing indices of
            vertices
        """
        if reduced:
            return self.reduced_vertices
        return self.vertices

    def get_all_voronoi_regions(self, reduced=False) -> List:
        """
        Getter equivalent to scipy's sv.regions.

        Returns:
            An list of sublists (could have different lengths) that define which vertices belong to individual center
            points of points; the order is the same as for self.get_all_voronoi_centers()
        """
        if reduced:
            return self.reduced_regions
        return self.regions

    def get_dim(self):
        return self.get_all_voronoi_centers().shape[1]

    def get_reduced_vertices_regions(self, **kwargs) -> Tuple:
        # vertices
        original_vertices = self.get_all_voronoi_vertices(**kwargs)
        indexes = np.unique(original_vertices, axis=0, return_index=True)[1]
        new_vertices = np.array([original_vertices[index] for index in sorted(indexes)])

        print(new_vertices[:3], original_vertices[:3])

        # regions
        old2new = {old_i: which_row_is_k(new_vertices, old)[0] for old_i, old in enumerate(original_vertices)}
        old_regions = self.get_all_voronoi_regions()
        new_regions = copy(old_regions)
        for i, region in enumerate(old_regions):
            for j, el in enumerate(region):
                new_regions[i][j] = old2new[el]
        return new_vertices, new_regions

    def get_sets_of_shared_vertices(self, reduced=False) -> coo_array:
        """
        Get a sparse array containing sets of reduced vertices shared between points
        """
        pass

    def get_voronoi_adjacency(self, **kwargs) -> coo_array:
        return self._calculate_N_N_array(property="adjacency", **kwargs)

    def get_center_distances(self, **kwargs) -> coo_array:
        """
        For points that are Voronoi neighbours, calculate the distance between them and save them in a sparse NxN
        format.

        Returns:

        """
        return self._calculate_N_N_array(property="center_distances", **kwargs)

    def get_cell_borders(self, **kwargs) -> coo_array:
        """
        For points that are Voronoi neighbours, calculate the distance between them and save them in a sparse NxN
        format.

        Returns:

        """
        return self._calculate_N_N_array(property="border_len", **kwargs)

    def _calculate_center_distances(self, index_1: int, index_2: int):
        pass

    def _calculate_borders(self, index_1: int, index_2: int):
        pass

    def _calculate_N_N_array(self, property="adjacency", **kwargs):
        reduced_vertices, reduced_regions = self.get_reduced_vertices_regions()

        # prepare for adjacency matrix
        rows = []
        columns = []
        elements = []

        for index_tuple in combinations(list(range(len(reduced_regions))), 2):

            set_1 = set(reduced_regions[index_tuple[0]])
            set_2 = set(reduced_regions[index_tuple[1]])

            if len(set_1.intersection(set_2)) >= self.get_dim() - 1:
                rows.extend([index_tuple[0], index_tuple[1]])
                columns.extend([index_tuple[1], index_tuple[0]])
                if property == "adjacency":
                    elements.extend([True, True])
                elif property == "center_distances":
                    dist = self._calculate_center_distances(*index_tuple)
                    elements.extend([dist, dist])
                elif property == "border_len":
                    # here the points are the voronoi indices that both cells share
                    indices_border = list(set_1.intersection(set_2))
                    dist = self._calculate_borders()

                    v1 = reduced_vertices[indices_border[0]]
                    v2 = reduced_vertices[indices_border[1]]
                    dist = dist_on_sphere(v1, v2)[0]
                    elements.extend([dist, dist])
                else:
                    raise ValueError(f"Didn't understand the argument property={property}.")
        N = len(self.get_all_voronoi_centers())
        adj_matrix = coo_array((elements, (rows, columns)), shape=(N, N))
        return adj_matrix


class RotobjVoronoi(AbstractVoronoi):

    def __init__(self, my_array: NDArray):
        self.my_array = my_array
        self.spherical_voronoi = SphericalVoronoi(my_array, radius=1, threshold=10 ** -UNIQUE_TOL)
        try:
            self.spherical_voronoi.sort_vertices_of_regions()
        except TypeError:
            pass
        super().__init__()

    def _create_centers_vertices_regions(self) -> Tuple:
        return self.spherical_voronoi.points, self.spherical_voronoi.vertices, self.spherical_voronoi.regions

    def get_cell_volumes(self, approx=False, using_detailed_grid=True, only_upper=False) -> NDArray:
        """
        From Voronoi cells you may also calculate areas on the sphere that are closest each grid point. The order of
        areas is the same as the order of points in self.grid. In Hyperspheres, these are volumes and only the
        approximation method is possible.

        Approximations only available for Ico, Cube3D and Cube4D polytopes.

        If only upper, the calculation will be done for all but only the results at upper indices will be returned.
        """
        dimensions = self.get_all_voronoi_centers().shape[1] # dimensionality of the space in which we position points
        if not approx and dimensions == 4:
            print("In 4D, only approximate calculation of Voronoi cells is possible. Proceeding numerically.")
            approx =True

        if not approx and dimensions == 3:
            output_array = np.array(self.spherical_voronoi.calculate_areas())
        elif approx:
            all_estimated_areas = []
            all_vertices = self.get_all_voronoi_vertices(only_upper=False)

            # for more precision, assign detailed grid points to closest sphere points
            # the first time, this will take a long time, but then, a saved version will be used
            if using_detailed_grid:
                if dimensions == 3:
                    dense_points = SphereGrid3DFactory.create("ico", N=2562, use_saved=True).get_grid_as_array(
                        only_upper=False)
                elif dimensions == 4:
                    dense_points = SphereGrid4DFactory.create("cube4D", N=272, use_saved=True).get_grid_as_array(
                        only_upper=False)
                else:
                    raise ValueError("Dimensions must be 3 or 4")
                extra_points_belongings = np.argmin(cdist(dense_points, self.get_all_voronoi_centers(only_upper=False),
                                                          metric="cos"), axis=1)

            for point_index, point in enumerate(self.get_all_voronoi_centers(only_upper=False)):
                vertices = all_vertices[point_index]
                if using_detailed_grid:
                    region_vertices_and_point = np.vstack(
                        [dense_points[extra_points_belongings == point_index], vertices])
                else:
                    region_vertices_and_point = np.vstack([point, vertices])
                my_convex_hull = ConvexHull(region_vertices_and_point, qhull_options='QJ')
                all_estimated_areas.append(my_convex_hull.area / 2)
            output_array = np.array(all_estimated_areas)
        else:
            print(f"Calculation and/or estimation of volume not possible. Returning None.")
            return

        if only_upper:
            return output_array[self.get_upper_indices()]
        else:
            return output_array

    def _calculate_center_distances(self, index_1: int, index_2: int):
        v1 = self.get_all_voronoi_centers()[index_1]
        v2 = self.get_all_voronoi_centers()[index_2]
        if self.get_dim() == 3:
            dist = dist_on_sphere(v1, v2)[0]
        else:
            dist = distance_between_quaternions(v1, v2)
        return dist

    def _calculate_N_N_array(self, property="adjacency", only_upper=False, include_opposing_neighbours=False):
        if self.dimensions == 3:
            N = self.get_N()
        else:
            N = 2*self.get_N()
        adj_matrix = super()._calculate_N_N_array(property=property).toarray() # todo: avoid non-sparse

        # TODO: here deal with only_upper, include_opposing_neighbours
        #adj_matrix = coo_array((elements, (rows, columns)), shape=(N, N)).toarray()
        if include_opposing_neighbours:

            all_grid = self.get_all_voronoi_centers(only_upper=False)
            ind2opp_index = dict()
            for d, n in enumerate(all_grid):
                inverse_el = find_inverse_quaternion(n)
                opp_ind = which_row_is_k(all_grid, inverse_el)
                if opp_ind:
                    ind2opp_index[d] = opp_ind[0]
            for i, line in enumerate(adj_matrix):
                for j, el in enumerate(line):
                    if el and j in ind2opp_index.keys():
                        adj_matrix[i][ind2opp_index[j]] = adj_matrix[i][j]
        if only_upper:
            available_indices = self._get_upper_indices()
            # Create a new array with the same shape as the original array
            extracted_arr = np.empty_like(adj_matrix, dtype=float)
            extracted_arr[:] = np.nan

            # Extract the specified rows and columns from the original array
            extracted_arr[available_indices, :] = adj_matrix[available_indices, :]
            extracted_arr[:, available_indices] = adj_matrix[:, available_indices]
            adj_matrix = extracted_arr

            # exclude nans
            valid_rows = np.all(~np.isnan(adj_matrix), axis=1)
            valid_columns = np.all(~np.isnan(adj_matrix), axis=0)

            # Extract the valid rows and columns from the original array
            extracted_arr = adj_matrix[valid_rows, :]
            extracted_arr = extracted_arr[:, valid_columns]
            adj_matrix = extracted_arr
            N = N // 2


class HalfRotobjVoronoi(RotobjVoronoi):


    def _get_upper_indices(self):
        """
        Get only the indices of full hypersphere array
        Returns:

        """
        upper_indices = [i for i, point in enumerate(self.my_array) if q_in_upper_sphere(point)]
        return sorted(upper_indices)

    def _create_centers_vertices_regions(self) -> Tuple:
        all_centers = self.spherical_voronoi.points
        all_vertices = self.spherical_voronoi.vertices
        all_regions = self.spherical_voronoi.regions

        half_centers = np.array([center for i, center in enumerate(all_centers) if i in self._get_upper_indices()])

        half_regions = [region for i, region in enumerate(all_regions) if i in self._get_upper_indices()]
        flattened_upper_regions = [x for sublist in half_regions for x in sublist]
        allowed_vertices = np.unique(flattened_upper_regions)
        half_vertices = np.array([vertex for i, vertex in enumerate(all_vertices) if i in allowed_vertices])

        # now update indices in regions
        new2old = {new_i: which_row_is_k(all_vertices, new)[0] for new_i, new in enumerate(half_vertices)}
        old2new = {o: n for n, o in new2old.items()}
        new_regions = []
        for i, point in enumerate(all_centers):
            if k_is_a_row(half_centers, point):
                fresh_list = [old2new[k] for k in all_regions[i] if k in old2new]
                new_regions.append(fresh_list)
        return half_centers, half_vertices, new_regions




class PositionVoronoi(AbstractVoronoi):

    """
    This object consists of several layers of spherical voronoi cells one above the other in several shells
    """

    def __init__(self, o_grid: NDArray, point_radii: NDArray):
        self.o_grid = o_grid
        self.point_radii = point_radii
        self.n_o = len(self.o_grid)
        self.n_t = len(self.point_radii)
        self.unit_sph_voronoi = SphericalVoronoi(self.o_grid, radius=1, threshold=10 ** -UNIQUE_TOL)

    def get_voronoi_radii(self):
        return get_between_radii(self.point_radii, include_zero=True)

    def _change_voronoi_radius(self, sv: SphericalVoronoi, new_radius: float) -> SphericalVoronoi:
        """
        This is a helper function. Since a FullGrid consists of several layers of spheres in which the points are at
        exactly same places (just at different radii), it makes sense not to recalculate, but just to scale the radius,
        vertices and points out of which the SphericalVoronoi consists to a new radius.
        """
        # important that it's a copy!
        sv = copy(sv)
        sv.radius = new_radius
        sv.vertices = normalise_vectors(sv.vertices, length=new_radius)
        sv.points = normalise_vectors(sv.points, length=new_radius)
        return sv

    def get_all_voronoi_centers(self) -> NDArray:
        """
        Get a position grid that is not structured layer-by-layer but is simply a 2D array of shape (N_t*N_o, 3) where
        N_t is the length of translation grid and N_o the length of orientation grid.

        Centers occur at radii self.point_radii
        """

        all_sv = [self._change_voronoi_radius(self.unit_sph_voronoi, r) for r in self.point_radii]
        return np.concatenate([sv.points for sv in all_sv])

    def get_all_voronoi_vertices(self) -> NDArray:
        """
        Get an array in which every element is a voronoi vertex, starting with the ones with smallest norm and then
        going towards larger radii.

        Vertices occur at radii self.get_voronoi_radii()
        """
        all_sv = [self._change_voronoi_radius(self.unit_sph_voronoi, r) for r in self.get_voronoi_radii()]
        # TODO: remove redundant vertices
        return np.concatenate([sv.vertices for sv in all_sv])

    def get_all_voronoi_regions(self) -> List:
        # first get regions associated with points in the same layer
        base_regions = self.unit_sph_voronoi.regions
        # vertices that in the end belong to a point are the "base points" in layers above and below
        num_base_vertices = len(self.unit_sph_voronoi.vertices)

        all_regions = []

        voronoi_points = self.get_all_voronoi_centers()
        for point_i, point in enumerate(voronoi_points):
            point_regions = []
            layer = point_i // self.n_o
            index_within_layer = point_i % self.n_o
            base_region = base_regions[index_within_layer]
            # points below
            point_regions.extend([x+layer*num_base_vertices for x in base_region])
            # points above
            point_regions.extend([x + (layer + 1) * num_base_vertices for x in base_region])
            all_regions.append(point_regions)
        return all_regions


