"""
Grids on N-dimensional spheres.

This module is one of central building blocks of the molgri package. The central object here is the SphereGridNDim
(And the corresponding Factory object that creates a sphere grid given the algorithm, dimension and number of points).
A SphereGridNDim object implements a grid that consists of points on a N-dimensional unit sphere.

Connected to:
 - polytope module - some algorithms for creating sphere grids are based of subdivision of polytopes
 - fullgrid module - combines a 3D SphereGridNDim, a 4D SphereGridNDim and a translation grid to sample the full
                     approach space
"""

import os
from abc import ABC, abstractmethod
from itertools import combinations
from time import time
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.sparse import coo_array
from scipy.spatial import SphericalVoronoi
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from molgri.assertions import which_row_is_k
from molgri.space.analysis import prepare_statistics, write_statistics
from molgri.space.utils import random_quaternions, random_sphere_points, \
    unique_quaternion_set, dist_on_sphere
from molgri.constants import UNIQUE_TOL, EXTENSION_GRID_FILES, NAME2PRETTY_NAME, SMALL_NS
from molgri.paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_STAT
from molgri.space.polytopes import Cube4DPolytope, IcosahedronPolytope, Cube3DPolytope, Polytope
from molgri.space.rotations import grid2rotation, rotation2grid4vector
from molgri.wrappers import time_method, save_or_use_saved, deprecated


SPHERE_SURFACE = 4*pi
HALF_HYPERSPHERE_SURFACE = pi**2


class SphereGridNDim(ABC):
    """
    This is a general and abstract implementation of a spherical grid in any number of dimensions (currently only
    using 3 or 4 dimensions). Each subclass is a particular implementation that must implement the abstract method
    _gen_grid_4D. If the implementation of _gen_grid_3D is not overridden, the 3D grid will be created from the 4D
    one using 4D grid to act as rotational quaternions on the unit z vector.
    """

    algorithm_name = "generic"

    def __init__(self, dimensions: int, N: int = None, use_saved: bool = True,
                 time_generation: bool = False):
        self.dimensions = dimensions
        self.N = N
        self.gen_algorithm = self.algorithm_name
        self.use_saved = use_saved
        self.time_generation = time_generation
        self.grid: Optional[NDArray] = None
        self.spherical_voronoi: Optional[SphericalVoronoi] = None
        self.polytope = None

    def __len__(self) -> int:
        return self.get_N()

    def get_N(self) -> int:
        """Get the number of points in the self.grid array. It is important to use this getter and not the attribute
        self.N as they can differ if the self.filter_non_unique option is set."""
        return len(self.get_grid_as_array())

    def __str__(self):
        return f"Object {type(self).__name__} <{self.get_decorator_name()}>"

    ##################################################################################################################
    #                      generation/loading of grids
    ##################################################################################################################

    def gen_grid(self) -> NDArray:
        """
        This method saves to self.grid if it has been None before (and checks the format) but returns nothing.
        This method only implements loading/timing/printing logic, the actual process of creation is outsourced to
        self._gen_grid() that is implemented by sub-classes.

        To get the grid, use self.get_grid_as_array().
        """
        # condition that there is still something to generate
        if self.grid is None:
            if self.time_generation:
                self.grid = self.gen_and_time()
            else:
                self.grid = self._gen_grid()
        assert isinstance(self.grid, np.ndarray), "A grid must be a numpy array!"
        assert self.grid.shape == (self.N, self.dimensions), f"Grid not of correct shape!"
        assert np.allclose(np.linalg.norm(self.grid, axis=1), 1, atol=10 ** (-UNIQUE_TOL)), "A grid must have norm 1!"
        return self.grid

    def get_grid_as_array(self):
        return self.grid

    @abstractmethod
    def _gen_grid(self) -> NDArray:
        raise NotImplementedError(f"Generating sphere grids with {self.algorithm_name} was not implemented!")

    @time_method
    def gen_and_time(self) -> NDArray:
        """Same as generation, but prints the information about time needed. Useful for end users."""
        return self._gen_grid()


    ##################################################################################################################
    #                      name and path getters
    ##################################################################################################################

    def get_name(self, with_dim: bool = True) -> str:
        """
        This is a standard name that can be used for saving files, but not pretty enough for titles etc.

        Args:
            with_dim: if True, include _3d or _4d to include dimensionality of the grid

        Returns:
            a string usually of the form 'ico_17'
        """
        output = f"{self.gen_algorithm}"
        if self.N is not None:
            output += f"_{self.N}"
        if with_dim:
            output += f"_{self.dimensions}d"
        return output

    def get_decorator_name(self) -> str:
        """Name used in printing and decorators, not suitable for file names."""
        return f"{NAME2PRETTY_NAME[self.gen_algorithm]} algorithm, {self.N} points"

    def get_grid_path(self, extension=EXTENSION_GRID_FILES) -> str:
        """Get the entire path to where the fullgrid is saved."""
        return f"{PATH_OUTPUT_ROTGRIDS}{self.get_name(with_dim=True)}.{extension}"

    def get_statistics_path(self, extension) -> str:
        """get the entire path to where the statistics are saved."""
        return f"{PATH_OUTPUT_STAT}{self.get_name(with_dim=True)}.{extension}"

    ##################################################################################################################
    #                      useful methods
    ##################################################################################################################

    def save_uniformity_statistics(self, num_random: int = 100, alphas=None):
        """
        Deprecated - all necessary data is saved by individual methods using save_or_use_saved.
        Save both the short, user-friendly summary of statistics and the entire uniformity data.
        """
        short_statistics_path = self.get_statistics_path(extension="txt")
        statistics_path = self.get_statistics_path(extension="csv")
        stat_data, full_data = prepare_statistics(self.get_grid_as_array(), alphas, d=self.dimensions,
                                                  num_rand_points=num_random)
        write_statistics(stat_data, full_data, short_statistics_path, statistics_path,
                         num_random, name=self.get_name(), dimensions=self.dimensions)

    @save_or_use_saved
    def get_uniformity_df(self, alphas):
        """
        Get the dataframe necessary to draw violin plots showing how uniform different generation algorithms are.
        """
        # recalculate if: 1) self.use_saved_data = False OR 2) no saved data exists
        if not self.use_saved or not os.path.exists(self.get_statistics_path("csv")):
            self.save_uniformity_statistics(alphas=alphas)
        ratios_df = pd.read_csv(self.get_statistics_path("csv"), dtype=float)
        # OR 3) provided alphas don't match those in the found file
        saved_alphas = set(ratios_df["alphas"])
        if saved_alphas != alphas:
            self.save_uniformity_statistics(alphas=alphas)
            ratios_df = pd.read_csv(self.get_statistics_path("csv"), dtype=float)
        return ratios_df

    @save_or_use_saved
    def get_convergence_df(self, alphas: tuple, N_list: tuple = None):
        """
         Get the dataframe necessary to draw convergence plots for various values of N.
         """
        if N_list is None:
            # create equally spaced convergence set
            assert self.N >= 3, f"N={self.N} not large enough to study convergence"
            N_list = np.logspace(np.log10(3), np.log10(self.N), dtype=int)
            N_list = np.unique(N_list)
        full_df = []
        for N in N_list:
            if self.dimensions == 3:
                grid_factory = SphereGrid3DFactory
            else:
                grid_factory = SphereGrid4DFactory
            grid_factory = grid_factory.create(alg_name=self.gen_algorithm, N=N, time_generation=False,
                                                    use_saved=self.use_saved)
            df = grid_factory.get_uniformity_df(alphas=alphas)
            df["N"] = len(grid_factory.get_grid_as_array())
            full_df.append(df)
        full_df = pd.concat(full_df, axis=0, ignore_index=True)
        return full_df

    @save_or_use_saved
    def get_spherical_voronoi_cells(self):
        """
        A spherical grid (in 3D) can be used as a basis for a spherical Voronoi grid. In this case, each grid point is
        used as a center of a Voronoi cell. The spherical Voronoi cells also have the radius 1.

        The division into cells will fail in case points are not unique (too close to each other)
        """
        if self.spherical_voronoi is None:
            self.spherical_voronoi = SphericalVoronoi(self.get_grid_as_array(), radius=1, threshold=10**-UNIQUE_TOL)
        return self.spherical_voronoi

    def _get_reduced_sv_vertices(self) -> NDArray:
        original_vertices = self.get_spherical_voronoi_cells().vertices
        indexes = np.unique(original_vertices, axis=0, return_index=True)[1]
        return np.array([original_vertices[index] for index in sorted(indexes)])

    def _get_reduced_sv_regions(self):
        original_vertices = self.get_spherical_voronoi_cells().vertices
        new_vertices = self._get_reduced_sv_vertices()

        old_index2new_index = dict()
        for original_i, original_point in enumerate(original_vertices):
            new_index_opt = which_row_is_k(new_vertices, original_point)
            assert len(new_index_opt) == 1
            old_index2new_index[original_i] = new_index_opt[0]

        old_regions = self.spherical_voronoi.regions
        new_regions = []
        for region in old_regions:
            new_region = [old_index2new_index[oi] for oi in region]
            new_regions.append(new_region)

        return new_regions


    @abstractmethod
    @save_or_use_saved
    def get_cell_volumes(self, approx=False, using_detailed_grid=True) -> NDArray:
        """
        From Voronoi cells you may also calculate areas on the sphere that are closest each grid point. The order of
        areas is the same as the order of points in self.grid. In Hyperspheres, these are volumes and only the
        approximation method is possible.

        Approximations only available for Ico, Cube3D and Cube4D polytopes.
        """
        pass

    @abstractmethod
    def get_voronoi_adjacency(self):
        pass # TODO


class SphereGrid3Dim(SphereGridNDim, ABC):

    algorithm_name = "generic_3d"

    def __init__(self, N: int = None, use_saved: bool = True, time_generation: bool = False):
        super().__init__(dimensions=3, N=N, use_saved=use_saved, time_generation=time_generation)


    def get_cell_volumes(self, approx=False, using_detailed_grid=True) -> NDArray:
        """
        From Voronoi cells you may also calculate areas on the sphere that are closest each grid point. The order of
        areas is the same as the order of points in self.grid. In Hyperspheres, these are volumes and only the
        approximation method is possible.

        Approximations only available for Ico, Cube3D and Cube4D polytopes.
        """
        if self.polytope and approx:
            all_estimated_areas = []
            voronoi_cells = self.get_spherical_voronoi_cells()
            all_vertices = [voronoi_cells.vertices[region] for region in voronoi_cells.regions]

            # for more precision, assign detailed grid points to closest sphere points
            # the first time, this will take a long time, but then, a saved version will be used
            if using_detailed_grid:
                if self.algorithm_name == "ico":
                    dense_points = SphereGrid3DFactory.create("ico", N=2562, use_saved=True).get_grid_as_array()
                elif self.algorithm_name == "cube3D":
                    dense_points = SphereGrid3DFactory.create("cube3D", N=1538, use_saved=True).get_grid_as_array()
                else:
                    raise ValueError("Wrong algorithm choice to estimate Voronoi areas in 3D: try cube3D, ico or set "
                                     "approx=False")
                extra_points_belongings = np.argmin(cdist(dense_points, self.get_grid_as_array(), metric="cos"), axis=1)

            for point_index, point in enumerate(self.get_grid_as_array()):
                vertices = all_vertices[point_index]
                if using_detailed_grid:
                    region_vertices_and_point = np.vstack([dense_points[extra_points_belongings == point_index], vertices])
                else:
                    region_vertices_and_point = np.vstack([point, vertices])
                my_convex_hull = ConvexHull(region_vertices_and_point, qhull_options='QJ')
                all_estimated_areas.append(my_convex_hull.area / 2)
            return np.array(all_estimated_areas)
        elif not approx:
            sv = self.get_spherical_voronoi_cells()
            return np.array(sv.calculate_areas())
        else:
            return np.array([])

    def get_voronoi_adjacency(self) -> coo_array:
        return self._calculate_N_N_array(property="adjacency")

    def get_center_distances(self) -> coo_array:
        """
        For points that are Voronoi neighbours, calculate the distance between them and save them in a sparse NxN
        format.

        Returns:

        """
        return self._calculate_N_N_array(property="center_distances")

    def get_cell_borders(self) -> coo_array:
        """
        For points that are Voronoi neighbours, calculate the distance between them and save them in a sparse NxN
        format.

        Returns:

        """
        return self._calculate_N_N_array(property="border_len")

    def _calculate_N_N_array(self, property="adjacency"):
        #sv = self.get_spherical_voronoi_cells()
        reduced_vertices = self._get_reduced_sv_vertices()
        reduced_regions = self._get_reduced_sv_regions()
        N = self.get_N()
        points = self.get_grid_as_array()
        # prepare for adjacency matrix
        rows = []
        columns = []
        elements = []

        for index_tuple in combinations(list(range(len(reduced_regions))), 2):

            set_1 = set(reduced_regions[index_tuple[0]])
            set_2 = set(reduced_regions[index_tuple[1]])

            if len(set_1.intersection(set_2)) == 2:
                rows.extend([index_tuple[0], index_tuple[1]])
                columns.extend([index_tuple[1], index_tuple[0]])
                if property == "adjacency":
                    elements.extend([True, True])
                elif property == "center_distances":
                    v1 = points[index_tuple[0]]
                    v2 = points[index_tuple[1]]
                    dist = dist_on_sphere(v1, v2)[0]
                    elements.extend([dist, dist])
                elif property == "border_len":
                    # here the points are the voronoi indices that both cells share
                    indices_border = list(set_1.intersection(set_2))
                    v1 = reduced_vertices[indices_border[0]]
                    v2 = reduced_vertices[indices_border[1]]
                    dist = dist_on_sphere(v1, v2)[0]
                    elements.extend([dist, dist])
                else:
                    raise ValueError(f"Didn't understand the argument property={property}.")
        return coo_array((elements, (rows, columns)), shape=(N, N))


class SphereGrid4Dim(SphereGridNDim, ABC):
    algorithm_name = "generic_4d"

    def __init__(self, N: int = None, use_saved: bool = True, time_generation: bool = False):
        super().__init__(dimensions=4, N=N, use_saved=use_saved, time_generation=time_generation)
        self.full_hypersphere_grid = None

    def get_cell_volumes(self, approx=True, using_detailed_grid=True) -> NDArray:
        """
        From Voronoi cells you may also calculate areas on the sphere that are closest each grid point. The order of
        areas is the same as the order of points in self.grid. In Hyperspheres, these are volumes and only the
        approximation method is possible.

        Approximations only available for Ico, Cube3D and Cube4D polytopes.
        """
        if not approx:
            print("In 4D, only approximate calculation of Voronoi cells is possible. Proceeding numerically.")
        if not self.polytope:
            print("In 4D, only calculation of Voronoi cell volumes is only possible for polyhedra-based grids.")
            return np.array([])

        all_estimated_areas = []
        voronoi_cells = self.get_spherical_voronoi_cells()
        all_vertices = [voronoi_cells.vertices[region] for region in voronoi_cells.regions]

        # for more precision, assign detailed grid points to closest sphere points
        if using_detailed_grid:
            dense_points = SphereGrid4DFactory.create("cube4D", N=272, use_saved=True).get_grid_as_array()
            extra_points_belongings = np.argmin(cdist(dense_points, self.get_grid_as_array(), metric="cos"), axis=1)

        for point_index, point in enumerate(self.get_grid_as_array()):
            vertices = all_vertices[point_index]
            if using_detailed_grid:
                region_vertices_and_point = np.vstack([dense_points[extra_points_belongings == point_index], vertices])
            else:
                region_vertices_and_point = np.vstack([point, vertices])
            my_convex_hull = ConvexHull(region_vertices_and_point, qhull_options='QJ')
            all_estimated_areas.append(my_convex_hull.area / 2)
        return np.array(all_estimated_areas)

    def get_voronoi_adjacency(self):
        pass

    def get_full_hypersphere_array(self) -> NDArray:
        return self.full_hypersphere_grid


class ZeroRotations3D(SphereGrid3Dim):
    algorithm_name = "zero3D"
    def _gen_grid(self) -> NDArray:
        self.N = 1
        z_vec = np.array([[0, 0, 1]])
        return z_vec

class ZeroRotations4D(SphereGrid4Dim):
    algorithm_name = "zero4D"
    def _gen_grid(self):
        self.N = 1
        rot_matrix = np.eye(3)
        rot_matrix = rot_matrix[np.newaxis, :]
        self.rotations = Rotation.from_matrix(rot_matrix)
        return self.rotations.as_quat()

class RandomQRotations(SphereGrid4Dim):

    algorithm_name = "randomQ"

    def _gen_grid(self) -> NDArray:
        np.random.seed(0)
        all_quaternions = random_quaternions(4*self.N)
        # now select those that are in the upper hemisphere
        unique_quaternions = unique_quaternion_set(all_quaternions)[:self.N]
        max_index = which_row_is_k(all_quaternions, unique_quaternions[-1])[0]
        self.full_hypersphere_grid = all_quaternions[:max_index]
        return random_quaternions(self.N)


class RandomSRotations(SphereGrid3Dim):

    algorithm_name = "randomS"

    def _gen_grid(self) -> NDArray:
        np.random.seed(0)
        return random_sphere_points(self.N)


# class ZeroRotations(SphereGridNDim):
#     algorithm_name = "zero"
#
#     def get_voronoi_areas(self, approx=False, using_detailed_grid=True) -> NDArray:
#         # since only 1 point, return full area of (hyper)/sphere
#         if self.dimensions == 3:
#             return np.array(SPHERE_SURFACE)
#         elif self.dimensions == 4:
#             return np.array(HALF_HYPERSPHERE_SURFACE)
#         else:
#             raise ValueError(f"Need 3 or 4 dimensions to work, not {self.dimensions}")
#
#     def get_voronoi_adjacency(self):
#         pass
#
#     def _gen_grid(self):
#         self.N = 1
#         rot_matrix = np.eye(3)
#         rot_matrix = rot_matrix[np.newaxis, :]
#         self.rotations = Rotation.from_matrix(rot_matrix)
#         return self.rotations.as_quat()


class Cube4DRotations(SphereGrid4Dim):
    algorithm_name = "cube4D"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polytope = Cube4DPolytope()

    def _gen_grid(self):
        while len(self.polytope.get_half_of_hypercube()) < self.N:
            self.polytope.divide_edges()
        return self.polytope.get_half_of_hypercube(N=self.N, projection=True)


class FullDivCube4DRotations(SphereGrid4Dim):
    algorithm_name = "fulldiv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        allowed_num_of_orientations = (8, 40, 272, 2080)
        if self.N not in allowed_num_of_orientations:
            raise ValueError("Only full subdivisions of cube4D are allowed")
        self.polytope = Cube4DPolytope()
        for i in range(allowed_num_of_orientations.index(self.N)):
            self.polytope.divide_edges()

    def _gen_grid(self) -> NDArray:
        return self.polytope.get_half_of_hypercube(projection=True)


class IcoAndCube3DRotations(SphereGrid3Dim):
    algorithm_name: str = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_z_projection_grid(self):
        "In this case, directly construct the 3D object"

    def _gen_grid(self):
        while len(self.polytope.get_nodes()) < self.N:
            self.polytope.divide_edges()
        ordered_points = self.polytope.get_nodes(N=self.N, projection=True)
        return ordered_points


class IcoRotations(IcoAndCube3DRotations):

    algorithm_name = "ico"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polytope = IcosahedronPolytope()


class Cube3DRotations(IcoAndCube3DRotations):
    algorithm_name = "cube3D"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polytope = Cube3DPolytope()


class SphereGridFactory:

    @classmethod
    def create(cls, alg_name: str, N: int, dimensions: int, **kwargs):
        if dimensions == 3:
            return SphereGrid3DFactory.create(alg_name=alg_name, N=N, **kwargs)
        elif dimensions == 4:
            return SphereGrid4DFactory.create(alg_name=alg_name, N=N, **kwargs)
        else:
            raise ValueError("Cannot generate sphere grid for dimensions not in (3, 4).")


class SphereGrid3DFactory:

    """
    This should be the only access point to all SphereGridNDim objects. Simply provide the generation algorithm name,
    the number of points and dimensions as well as optional arguments. The requested object will be returned.
    """

    @classmethod
    def create(cls, alg_name: str, N: int, **kwargs) -> SphereGrid3Dim:
        if alg_name == "randomS":
            selected_sub_obj = RandomSRotations(N=N, **kwargs)
        elif alg_name == "ico":
            selected_sub_obj = IcoRotations(N=N, **kwargs)
        elif alg_name == "cube3D":
            selected_sub_obj = Cube3DRotations(N=N, **kwargs)
        elif alg_name == "zero3D":
            selected_sub_obj = ZeroRotations3D(N=N, **kwargs)
        else:
            raise ValueError(f"The algorithm {alg_name} not familiar to QuaternionGridFactory.")
        selected_sub_obj.gen_grid()
        return selected_sub_obj

class SphereGrid4DFactory:

    """
    This should be the only access point to all SphereGridNDim objects. Simply provide the generation algorithm name,
    the number of points and dimensions as well as optional arguments. The requested object will be returned.
    """

    @classmethod
    def create(cls, alg_name: str, N: int, **kwargs) -> SphereGridNDim:
        if alg_name == "randomQ":
            selected_sub_obj = RandomQRotations(N=N, **kwargs)
        elif alg_name == "cube4D":
            selected_sub_obj = Cube4DRotations(N=N, **kwargs)
        elif alg_name == "fulldiv":
            selected_sub_obj = FullDivCube4DRotations(N=N, **kwargs)
        elif alg_name == "zero4D":
            selected_sub_obj = ZeroRotations4D(N=N, **kwargs)
        else:
            raise ValueError(f"The algorithm {alg_name} not familiar to SphereGrid4DFactory.")
        selected_sub_obj.gen_grid()
        return selected_sub_obj


class ConvergenceSphereGridFactory:

    """
    This is a central object for studying the convergence of SphereGridNDim objects with the number of dimensions.
    """

    def __init__(self, alg_name: str, dimensions: int, N_set = None, use_saved=True, **kwargs):
        if N_set is None:
            N_set = SMALL_NS
        self.N_set = N_set
        self.alg_name = alg_name
        self.dimensions = dimensions
        self.use_saved = use_saved
        self.kwargs = kwargs
        self.list_sphere_grids = []

    def get_name(self):
        N_min = self.N_set[0]
        N_max = self.N_set[-1]
        N_len = len(self.N_set)
        return f"convergence_{self.alg_name}_{self.dimensions}d_{N_min}_{N_max}_{N_len}"

    def create(self) -> list:
        list_sphere_grids = []
        for N in self.N_set:
            if self.dimensions == 3:
                sg = SphereGrid3DFactory.create(self.alg_name, N, use_saved=self.use_saved, **self.kwargs)
            else:
                sg = SphereGrid4DFactory.create(self.alg_name, N, use_saved=self.use_saved, **self.kwargs)
            list_sphere_grids.append(sg)
        return list_sphere_grids

    @save_or_use_saved
    def get_list_sphere_grids(self):
        if not self.list_sphere_grids:
            self.list_sphere_grids = self.create()
        return self.list_sphere_grids

    @save_or_use_saved
    def get_spherical_voronoi_areas(self):
        """
        Get plotting-ready data on the size of Voronoi areas for different numbers of points.
        """
        data = []
        for N, sg in zip(self.N_set, self.get_list_sphere_grids()):
            real_N = len(sg.get_grid_as_array())
            ideal_area = 4*pi/real_N
            try:
                real_areas = sg.get_cell_volumes()
            except ValueError:
                real_areas = [np.NaN] * real_N
            for area in real_areas:
                data.append([real_N, ideal_area, area])
        df = pd.DataFrame(data, columns=["N", "ideal area", "sph. Voronoi cell area"])
        return df

    @save_or_use_saved
    def get_generation_times(self, repeats=5):
        """
        Get plotting-ready data on time needed to generate spherical grids of different sizes. Each generation is
        repeated several times to be able to estimate the error.
        """
        data = []
        for N in self.N_set:
            for _ in range(repeats):
                t1 = time()
                # cannot use saved if you are timing!
                if self.dimensions == 3:
                    sg = SphereGrid3DFactory.create(self.alg_name, N, use_saved=self.use_saved, **self.kwargs)
                else:
                    sg = SphereGrid4DFactory.create(self.alg_name, N, use_saved=self.use_saved, **self.kwargs)
                t2 = time()
                data.append([len(sg.get_grid_as_array()), t2 - t1])
        df = pd.DataFrame(data, columns=["N", "Time [s]"])
        return df


