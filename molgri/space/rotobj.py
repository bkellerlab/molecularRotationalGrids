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

from abc import ABC, abstractmethod
from time import time
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial.transform import Rotation

from molgri.space.utils import (find_inverse_quaternion, random_quaternions, random_sphere_points,
                                hemisphere_quaternion_set, q_in_upper_sphere)
from molgri.constants import UNIQUE_TOL, NAME2PRETTY_NAME, SMALL_NS
from molgri.space.polytopes import Cube4DPolytope, IcosahedronPolytope, Cube3DPolytope
from molgri.space.voronoi import RotobjVoronoi, AbstractVoronoi, HalfRotobjVoronoi, MikroVoronoi


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
        self.polytope = None
        self.spherical_voronoi: Optional[AbstractVoronoi] = None

    def __getattr__(self, name):
        """ Enable forwarding methods to self.position_grid, so that from SphereGridNDim you can access all properties and
         methods of RotobjVoronoi too."""
        return getattr(self.spherical_voronoi, name)

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
        if self.dimensions == 3:
            assert self.grid.shape == (self.N, self.dimensions), f"3D Grid not of correct shape!"
        elif self.dimensions == 4:
            assert self.grid.shape == (2*self.N, self.dimensions), f"4D Grid (double coverage) not of correct shape!"
        assert np.allclose(np.linalg.norm(self.grid, axis=1), 1, atol=10 ** (-UNIQUE_TOL)), "A grid must have norm 1!"

        # the corresponding voronoi
        if self.dimensions == 3 and self.N >= 4:
            self.spherical_voronoi = RotobjVoronoi(self.grid)
        elif self.dimensions == 4 and self.N >= 4:
            self.spherical_voronoi = HalfRotobjVoronoi(self.grid)
        else:
            print("Warning! For <=4 points, volumes, areas etc are only estimated.")
            self.spherical_voronoi = MikroVoronoi(dimensions=self.dimensions, N_points=self.get_N())
        return self.grid

    def get_grid_as_array(self, only_upper: bool = False) -> NDArray:
        """
        Get a numpy array in which every row is a point on a (hyper)(half)sphere.

        Args:
            only_upper (bool): if True will only return (roughly) half of the points, namely those whose first
            non-zero component is positive

        Returns:
            an array of points
        """
        if only_upper:
            return self.grid[self.get_upper_indices()]
        else:
            return self.grid

    @abstractmethod
    def _gen_grid(self) -> NDArray:
        raise NotImplementedError(f"Generating sphere grids with {self.algorithm_name} was not implemented!")


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

    def get_spherical_voronoi(self):
        """
        A spherical grid (in 3D) can be used as a basis for a spherical Voronoi grid. In this case, each grid point is
        used as a center of a Voronoi cell. The spherical Voronoi cells also have the radius 1.

        The division into cells will fail in case points are not unique (too close to each other)
        """
        return self.spherical_voronoi

    def get_upper_indices(self):
        """
        Get only the indices of full hypersphere array
        Returns:

        """
        upper_indices = [i for i, point in enumerate(self.get_grid_as_array(only_upper=False)) if q_in_upper_sphere(
            point)]
        return sorted(upper_indices)


class SphereGrid3Dim(SphereGridNDim, ABC):

    algorithm_name = "generic_3d"

    def __init__(self, N: int = None, use_saved: bool = True, time_generation: bool = False):
        super().__init__(dimensions=3, N=N, use_saved=use_saved, time_generation=time_generation)


class SphereGrid4Dim(SphereGridNDim, ABC):
    algorithm_name = "generic_4d"

    def __init__(self, N: int = None, use_saved: bool = True, time_generation: bool = False):
        super().__init__(dimensions=4, N=N, use_saved=use_saved, time_generation=time_generation)

    def _gen_grid(self) -> NDArray:
        half_grid = self.grid
        N = self.N
        full_hypersphere_grid = np.zeros((2 * N, 4))
        full_hypersphere_grid[:N] = half_grid
        for i in range(N):
            inverse_q = find_inverse_quaternion(half_grid[i])
            full_hypersphere_grid[N + i] = inverse_q
        self.grid = full_hypersphere_grid
        return self.grid

    def get_grid_as_array(self, only_upper: bool = True) -> NDArray:
        """
        By default, get an array of shape (N, 4) where every row is a quaternion in the upper hemisphere of a
        hypersphere.

        Args:
            only_upper: if False, return an array of shape (2N, 4), where you first get N quaternions on upper
            sphere and then also get their N exactly opposing points on bottom hemisphere.
        """
        return super(SphereGrid4Dim, self).get_grid_as_array(only_upper=only_upper)


class ZeroRotations3D(SphereGrid3Dim):
    algorithm_name = "zero3D"
    def _gen_grid(self) -> NDArray:
        self.N = 1
        z_vec = np.array([[0, 0, 1]])
        return z_vec

# class HandMade3D(SphereGrid3Dim):
#     """
#     This is for tests - generate a SphereGrid3Dim object from any list or array. Fails if not 3D or not points on
#     unit sphere.
#     """
#     algorithm_name = "handmade3D"
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if "points" not in kwargs.keys:
#             raise ValueError("HandMade grids must provide an argument points.")
#         self.points = np.array(kwargs["points"])
#
#     def _gen_grid(self) -> NDArray:
#
#         assert all_row_norms_equal_k
#         self.N = len(self.points)
#         return self.points


class ZeroRotations4D(SphereGrid4Dim):
    algorithm_name = "zero4D"

    def _gen_grid(self):
        self.N = 1
        rot_matrix = np.eye(3)
        rot_matrix = rot_matrix[np.newaxis, :]
        self.rotations = Rotation.from_matrix(rot_matrix)
        self.grid = self.rotations.as_quat()
        return super()._gen_grid()


class RandomQRotations(SphereGrid4Dim):

    algorithm_name = "randomQ"

    def _gen_grid(self) -> NDArray:
        np.random.seed(0)
        all_quaternions = random_quaternions(self.N)
        # now select those that are in the upper hemisphere
        self.grid = hemisphere_quaternion_set(all_quaternions)
        return super()._gen_grid()


class RandomSRotations(SphereGrid3Dim):

    algorithm_name = "randomS"

    def _gen_grid(self) -> NDArray:
        np.random.seed(0)
        return random_sphere_points(self.N)

class Cube4DRotations(SphereGrid4Dim):
    algorithm_name = "cube4D"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polytope = Cube4DPolytope()

    def _gen_grid(self):
        while len(self.polytope.get_half_of_hypercube()) < self.N:
            self.polytope.divide_edges()
        self.grid = self.polytope.get_half_of_hypercube(N=self.N, projection=True)
        return super()._gen_grid()


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
        self.grid = self.polytope.get_half_of_hypercube(projection=True)
        return super()._gen_grid()


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
    def create(cls, alg_name: str, N: int, **kwargs) -> SphereGrid4Dim:
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

    def get_list_sphere_grids(self):
        if not self.list_sphere_grids:
            self.list_sphere_grids = self.create()
        return self.list_sphere_grids

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


