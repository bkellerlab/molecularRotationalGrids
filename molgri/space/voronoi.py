"""
Building up on scipy objects Voronoi and SphericalVoronoi

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.linalg import svd
from scipy.sparse import coo_array
from scipy.spatial import SphericalVoronoi, ConvexHull, Delaunay
from scipy.spatial.distance import cdist

from molgri.constants import UNIQUE_TOL
from molgri.space.utils import (dist_on_sphere, distance_between_quaternions,
                                exact_area_of_spherical_polygon, k_is_a_row, normalise_vectors, q_in_upper_sphere,
                                random_quaternions,
                                random_sphere_points, sort_points_on_sphere_ccw, which_row_is_k)


class AbstractVoronoi(ABC):

    """
    This is a general class I use as an extension of scipy Voronoi class. At the minimum, each subclass must
    implement getters for:
    1) centers
    2) vertices
    3) regions
    """

    def __init__(self, additional_points = None, **kwargs):
        created_vor = self._create_centers_vertices_regions()
        self.centers = copy(created_vor[0])
        self.vertices = copy(created_vor[1])
        self.regions = copy(created_vor[2])
        created_reduced = self.get_reduced_vertices_regions()
        self.reduced_vertices = copy(created_reduced[0])
        self.reduced_regions = copy(created_reduced[1])
        self.additional_points = additional_points

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

    def get_reduced_vertices_regions(self) -> Tuple:
        # vertices
        original_vertices = copy(self.get_all_voronoi_vertices(reduced=False))
        # correctly determines which lines to use
        indexes = np.unique(original_vertices, axis=0, return_index=True)[1]
        new_vertices = np.array([original_vertices[index] for index in sorted(indexes)])

        for old in original_vertices:
            k = which_row_is_k(new_vertices, old)
            if k is None:
                print(old)
        # regions
        # correctly assigns
        old2new = {old_i: which_row_is_k(new_vertices, old)[0] for old_i, old in enumerate(original_vertices)}
        old_regions = self.get_all_voronoi_regions()
        new_regions = []
        for i, region in enumerate(old_regions):
            fresh_region = []
            for j, el in enumerate(region):
                fresh_region.append(old2new[el])
            new_regions.append(fresh_region)
        return new_vertices, new_regions

    def _additional_points_per_cell(self) -> List:
        """
        Returns a list (len=num of centers) of lists in which there are additional points that are associated with
        each region (or empty list if there are none)
        """
        all_points = self.get_all_voronoi_centers()
        if self.additional_points is not None:
            result = []
            extra_points_belongings = np.argmin(cdist(self.additional_points, all_points,
                                                      metric="cos"), axis=1)
            for i, _ in enumerate(all_points):
                result.append(self.additional_points[extra_points_belongings == i])
            return result
        else:
            return [[] for _ in range(len(all_points))]

    def get_convex_hulls(self):
        """
        In the same order as self.get_all_centers(), get convex hulls using the point, the vertices and possibly
        additional points that belong into this region. Used for approximating volumes, areas ...
        """
        all_points = self.get_all_voronoi_centers()
        all_vertices = self.get_all_voronoi_vertices(reduced=True)
        all_regions = self.get_all_voronoi_regions(reduced=True)
        additional_assignments = self._additional_points_per_cell()
        all_hulls = []
        # in first approximation, take the volume of convex hull of vertices belonging to a point
        for i, region in enumerate(all_regions):
            # In the region there are vertices, central point and possibly additional point
            associated_vertices = all_vertices[region]
            within_region = np.vstack([all_points[i], associated_vertices])
            if np.any(additional_assignments[i]):
                within_region = np.vstack([additional_assignments[i], within_region])
            my_convex_hull = ConvexHull(within_region, qhull_options='QJ')
            all_hulls.append(my_convex_hull)
        return np.array(all_hulls)

    def get_voronoi_volumes(self, **kwargs) -> NDArray:
        all_hulls = self.get_convex_hulls()
        return np.array([my_convex_hull.area / 2 for my_convex_hull in all_hulls])

    def get_voronoi_adjacency(self, **kwargs) -> coo_array:
        return self._calculate_N_N_array(sel_property="adjacency", **kwargs)

    def get_center_distances(self, **kwargs) -> coo_array:
        """
        For points that are Voronoi neighbours, calculate the distance between them and save them in a sparse NxN
        format.

        Returns:

        """
        return self._calculate_N_N_array(sel_property="center_distances", **kwargs)

    def get_cell_borders(self, **kwargs) -> coo_array:
        """
        For points that are Voronoi neighbours, calculate the distance between them and save them in a sparse NxN
        format.

        Returns:

        """
        return self._calculate_N_N_array(sel_property="border_len", **kwargs)

    def _calculate_center_distances(self, index_1: int, index_2: int):
        pass

    def _calculate_borders(self, index_1: int, index_2: int):
        pass

    def _calculate_N_N_array(self, sel_property="adjacency", **kwargs):
        reduced_regions = self.get_all_voronoi_regions(reduced=True)

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
                if sel_property == "adjacency":
                    elements.extend([True, True])
                elif sel_property == "center_distances":
                    dist = self._calculate_center_distances(*index_tuple)
                    elements.extend([dist, dist])
                elif sel_property == "border_len":
                    dist = self._calculate_borders(*index_tuple)
                    elements.extend([dist, dist])
                else:
                    raise ValueError(f"Didn't understand the argument property={sel_property}.")
        N = len(self.get_all_voronoi_centers())
        adj_matrix = coo_array((elements, (rows, columns)), shape=(N, N))
        return adj_matrix


class RotobjVoronoi(AbstractVoronoi):

    def __init__(self, my_array: NDArray, using_detailed_grid: bool = True):
        self.my_array = my_array
        self.using_detailed_grid = using_detailed_grid
        norm = np.linalg.norm(my_array, axis=1)[0]
        self.spherical_voronoi = SphericalVoronoi(my_array, radius=norm, threshold=10**-UNIQUE_TOL) #radius=,
        try:
            self.spherical_voronoi.sort_vertices_of_regions()
        except TypeError:
            pass
        dimensions = my_array.shape[1]
        np.random.seed(1)
        if using_detailed_grid and dimensions == 3:
            dense_points = normalise_vectors(random_sphere_points(3000), length=norm)
        elif using_detailed_grid and dimensions == 4:
            dense_points = random_quaternions(3000)
        else:
            dense_points = None
        super().__init__(additional_points=dense_points)

    def _create_centers_vertices_regions(self) -> Tuple:
        return self.spherical_voronoi.points, self.spherical_voronoi.vertices, self.spherical_voronoi.regions

    def get_voronoi_volumes(self, approx=False) -> Optional[NDArray]:
        """
        From Voronoi cells you may also calculate areas on the sphere that are closest each grid point. The order of
        areas is the same as the order of points in self.grid. In Hyperspheres, these are volumes and only the
        approximation method is possible.

        Approximations only available for Ico, Cube3D and Cube4D polytopes.

        If only upper, the calculation will be done for all but only the results at upper indices will be returned.
        """
        dimensions = self.get_dim() # dimensionality of the space in which we position points
        if not approx and dimensions == 4:
            print("In 4D, only approximate calculation of Voronoi cells is possible. Proceeding numerically.")
            approx = True
        if not approx and dimensions == 3:
            return np.array(self.spherical_voronoi.calculate_areas())
        elif approx:
            return super().get_voronoi_volumes()
        else:
            print(f"Calculation and/or estimation of volume not possible. Returning None.")
            return

    def _calculate_center_distances(self, index_1: int, index_2: int):
        v1 = self.get_all_voronoi_centers()[index_1]
        v2 = self.get_all_voronoi_centers()[index_2]
        if self.get_dim() == 3:
            dist = dist_on_sphere(v1, v2)
        else:
            dist = distance_between_quaternions(v1, v2)
        return dist

    def _calculate_borders(self, index_1: int, index_2: int):
        # here the points are the voronoi indices that both cells share
        reduced_vertices = self.get_all_voronoi_vertices(reduced=True)
        reduced_regions = self.get_all_voronoi_regions(reduced=True)
        set_1 = set(reduced_regions[index_1])
        set_2 = set(reduced_regions[index_2])
        indices_border = list(set_1.intersection(set_2))

        shared_vertices = reduced_vertices[indices_border]
        # matrix rank of shared_vertices should be one less than the dimensionality of space, because it's a
        # normal sphere (well, some points on a sphere) hidden inside 4D coordinates, or a part of a planar circle
        # expressed with 3D coordinates
        assert np.linalg.matrix_rank(shared_vertices) == self.get_dim() - 1
        u, s, vh = svd(shared_vertices)
        # rotate till last dimension is only zeros, then cut off the redundant dimension. Now we can correctly
        # calculate borders using lower-dimensional tools
        border_full_rank_points = np.dot(shared_vertices, vh.T)[:, :-1]
        # the points have unit norm

        # print(np.round(np.dot(shared_vertices, vh.T), 3))
        if self.get_dim() == 3:
            return dist_on_sphere(border_full_rank_points[0], border_full_rank_points[1])
        else:
            border_full_rank_points = sort_points_on_sphere_ccw(border_full_rank_points)
            area = exact_area_of_spherical_polygon(border_full_rank_points)
            return area

    def get_related_half_voronoi(self) -> HalfRotobjVoronoi:
        """
        The RotobjVoronoi deals with entire hyperspheres. If you are interested in filtering out only data applying to
        the upper part of this same grid, use this method

        Returns:
            HalfRotobjVoronoi object that only contains (roughly) half of the points of this grid that fall in the
            upper hemisphere. All getters of the Half object will return vertices, volumes ... only for these half
            points.
        """
        return HalfRotobjVoronoi(self.my_array, self.using_detailed_grid)


class HalfRotobjVoronoi(RotobjVoronoi):

    def __init__(self, my_array: NDArray, using_detailed_grid: bool = True):
        self.full_voronoi = RotobjVoronoi(my_array=my_array, using_detailed_grid=using_detailed_grid)
        super().__init__(my_array=my_array, using_detailed_grid=using_detailed_grid)

    def _additional_points_per_cell(self) -> List:
        """
        For half voronoi we need to filter additional points that are not in upper hypersphere
        """
        if self.additional_points is not None:
            self.additional_points = np.array([ap for ap in self.additional_points if q_in_upper_sphere(ap)])
        return super()._additional_points_per_cell()

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

    def get_voronoi_volumes(self, approx=False) -> Optional[NDArray]:
        """
        From Voronoi cells you may also calculate areas on the sphere that are closest each grid point. The order of
        areas is the same as the order of points in self.grid. In Hyperspheres, these are volumes and only the
        approximation method is possible.

        This cannot be done directly for half a sphere. We need the calculation on full sphere and then select only
        relevant volumes
        """
        all_volumes = self.full_voronoi.get_voronoi_volumes(approx=approx)
        return np.array([all_volumes[i] for i in self._get_upper_indices()])

    def _calculate_N_N_array(self, sel_property="adjacency", only_upper=True, include_opposing_neighbours=True):
        # warning! here use self.full_voronoi NOT super() to calculate the adj on full sphere
        adj_matrix = self.full_voronoi._calculate_N_N_array(sel_property=sel_property).toarray()

        if include_opposing_neighbours:
            all_grid = self.spherical_voronoi.points
            ind2opp_index = dict()
            for d, n in enumerate(all_grid):
                inverse_el = -n
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
        return coo_array(adj_matrix)


class MikroVoronoi(AbstractVoronoi):

    def _create_centers_vertices_regions(self) -> Tuple:
        pass

    def __init__(self, dimensions, N_points):
        self.dimensions = dimensions
        assert self.dimensions in [3, 4]
        self.N_points = N_points

    def get_voronoi_volumes(self, **kwargs) -> NDArray:
        if self.dimensions == 3:
            # area of unit sphere divided into  N parts
            return np.array([4*pi/self.N_points]*self.N_points)
        else:
            # hyperarea of half unit hypersphere  divided into  N parts
            return np.array([2 * pi**2 /2 / self.N_points] * self.N_points)

    def get_voronoi_adjacency(self, **kwargs) -> coo_array:
        result = np.eye(self.N_points)
        # invert 0s and 1s
        result =1 - result
        return coo_array(result)

    def get_center_distances(self, **kwargs) -> coo_array:
        """
        For points that are Voronoi neighbours, calculate the distance between them and save them in a sparse NxN
        format.

        Returns:

        """
        return self.get_voronoi_adjacency()

    def get_cell_borders(self, **kwargs) -> coo_array:
        """
        For points that are Voronoi neighbours, calculate the distance between them and save them in a sparse NxN
        format.

        Returns:

        """
        return self.get_voronoi_adjacency()


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
