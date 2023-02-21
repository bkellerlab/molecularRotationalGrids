import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial import SphericalVoronoi
from scipy.spatial.transform import Rotation

from molgri.space.analysis import prepare_statistics, write_statistics
from molgri.space.cells import surface_per_cell_ideal
from molgri.space.utils import random_quaternions, standardise_quaternion_set
from molgri.constants import UNIQUE_TOL, EXTENSION_GRID_FILES, NAME2PRETTY_NAME, SMALL_NS
from molgri.paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_STAT
from molgri.space.polytopes import Cube4DPolytope, IcosahedronPolytope, Cube3DPolytope, Polytope
from molgri.space.rotations import grid2rotation, rotation2grid4vector
from molgri.wrappers import time_method, save_or_use_saved


class SphereGridNDim(ABC):

    algorithm_name = "generic"

    def __init__(self, dimensions: int, N: int = None, use_saved: bool = True,
                 print_messages: bool = False, time_generation: bool = False):
        self.dimensions = dimensions
        self.N = N
        self.gen_algorithm = self.algorithm_name
        self.use_saved = use_saved
        self.time_generation = time_generation
        self.print_messages = print_messages
        self.grid: NDArray = None
        self.spherical_voronoi: SphericalVoronoi = None

    def __len__(self):
        return self.N

    def __str__(self):
        return f"Object {type(self).__name__} <{self.get_decorator_name()}>"

    ##################################################################################################################
    #                      generation/loading of grids
    ##################################################################################################################

    def gen_grid(self):
        """
        This method saves to self.grid if it has been None before (and checks the format) but returns nothing.
        This method only implements loading/timing/printing logic, the actual process of creation is outsourced to
        self._gen_grid() that is implemented by sub-classes.

        To get the grid, use self.get_grid_as_array().
        """
        # condition that there is still something to generate
        if self.grid is None:
            # if self.use_saved and the file exists, load it
            if self.use_saved and os.path.isfile(self.get_grid_path()):
                self.grid = np.load(self.get_grid_path())
            # else use the right generation function
            else:
                if self.time_generation:
                    self.grid = self.gen_and_time()
                else:
                    self.grid = self._gen_grid()
            if self.print_messages:
                self._check_uniformity()
        assert isinstance(self.grid, np.ndarray), "A grid must be a numpy array!"
        assert self.grid.shape == (self.N, self.dimensions), f"Grid not of correct shape!"
        assert np.allclose(np.linalg.norm(self.grid, axis=1), 1, atol=10 ** (-UNIQUE_TOL)), "A grid must have norm 1!"

    def _gen_grid(self) -> NDArray:
        if self.dimensions == 4:
            return self._gen_grid_4D()
        elif self.dimensions == 3:
            return self._gen_grid_3D()
        else:
            raise NotImplementedError(f"Generating sphere grids in {self.dimensions} not implemented!")

    def _gen_grid_3D(self) -> NDArray:
        """
        This is the default of obtaining 3D grid by projecting on z-vector, should be overwritten for ico and cube3D
        """
        quaternions = self._gen_grid_4D()
        rotations = Rotation.from_quat(quaternions)
        return rotation2grid4vector(rotations)

    @abstractmethod
    def _gen_grid_4D(self) -> NDArray:
        pass

    @time_method
    def gen_and_time(self) -> NDArray:
        return self._gen_grid()

    def _check_uniformity(self):
        as_quat = False
        if self.dimensions == 4:
            as_quat = True
        unique_grid = provide_unique(self.grid, as_quat=as_quat)
        if len(self.grid) != len(unique_grid):
            print(f"Warning! {len(self.grid) - len(unique_grid)} grid points of "
                  f"{self.get_name(with_dim=True)} non-unique (distance < 10^-{UNIQUE_TOL}).")

    ##################################################################################################################
    #                      name and path getters
    ##################################################################################################################

    def get_name(self, with_dim=False) -> str:
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
        return f"{NAME2PRETTY_NAME[self.gen_algorithm]} algorithm, {self.N} points"

    def get_grid_path(self, extension=EXTENSION_GRID_FILES) -> str:
        return f"{PATH_OUTPUT_ROTGRIDS}{self.get_name(with_dim=True)}.{extension}"

    def get_statistics_path(self, extension) -> str:
        return f"{PATH_OUTPUT_STAT}{self.get_name(with_dim=True)}.{extension}"

    ##################################################################################################################
    #                      useful methods
    ##################################################################################################################

    def get_grid_as_array(self) -> NDArray:
        """
        Get the sphere grid. If not yet created but N is set, will generate/load the grid automatically.
        """
        self.gen_grid()
        return self.grid

    def save_grid(self, extension: str = EXTENSION_GRID_FILES):
        if extension == "txt":
            # noinspection PyTypeChecker
            np.savetxt(self.get_grid_path(extension=extension), self.get_grid_as_array())
        else:
            np.save(self.get_grid_path(extension=extension), self.get_grid_as_array())

    def save_uniformity_statistics(self, num_random: int = 100, alphas=None):
        short_statistics_path = self.get_statistics_path(extension="txt")
        statistics_path = self.get_statistics_path(extension="csv")
        stat_data, full_data = prepare_statistics(self.get_grid_as_array(), alphas, d=self.dimensions,
                                                  num_rand_points=num_random)
        write_statistics(stat_data, full_data, short_statistics_path, statistics_path,
                         num_random, name=self.get_name(), dimensions=self.dimensions,
                         print_message=self.print_messages)

    @save_or_use_saved
    def get_uniformity_df(self, alphas):
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
        if N_list is None:
            # create equally spaced convergence set
            assert self.N >= 3, f"N={self.N} not large enough to study convergence"
            N_list = np.logspace(np.log10(3), np.log10(self.N), dtype=int)
            N_list = np.unique(N_list)
        full_df = []
        for N in N_list:
            grid_factory = SphereGridFactory.create(alg_name=self.gen_algorithm, N=N, dimensions=self.dimensions,
                                                    print_messages=False, time_generation=False,
                                                    use_saved=self.use_saved)
            df = grid_factory.get_uniformity_df(alphas=alphas)
            df["N"] = N
            full_df.append(df)
        full_df = pd.concat(full_df, axis=0, ignore_index=True)
        return full_df

    @save_or_use_saved
    def get_spherical_voronoi_cells(self):
        assert self.dimensions == 3, "Spherical voronoi cells only available for N=3"
        if self.spherical_voronoi is None:
            try:
                self.spherical_voronoi = SphericalVoronoi(self.grid, radius=1, threshold=10**-UNIQUE_TOL)
            except ValueError:
                if self.print_messages:
                    print(f"Spherical voronoi not created for {self.get_name()} due to duplicate generators")
                self.spherical_voronoi = []
        return self.spherical_voronoi

    @save_or_use_saved
    def get_voronoi_areas(self):
        sv = self.get_spherical_voronoi_cells()
        return sv.calculate_areas()


def _select_unique_rotations(rotations):
    rot_matrices = rotations.as_matrix()
    rot_matrices = np.unique(np.round(rot_matrices, UNIQUE_TOL), axis=0)
    return Rotation.from_matrix(rot_matrices)


def provide_unique(el_array: NDArray, tol: int = UNIQUE_TOL, as_quat=False) -> NDArray:
    """
    Take an array of any shape in which each row is an element. Return only the elements that are unique up to
    the tolerance.
    # TODO: maybe you could do some sort of decomposition to determine the point of uniqueness?

    Args:
        el_array: array with M elements of any shape
        tol: number of decimal points to consider
        as_quat: select True if el_array is an array of quatenions and you want to consider q and -q to be non-unique
                 (respecting double-coverage of quaternions)

    Returns:
        array with N <= M elements of el_array that are all unique
    """
    if as_quat:
        assert el_array.shape[1] == 4
        _, indices = np.unique(standardise_quaternion_set(el_array.round(tol)), axis=0, return_index=True)
    else:
        _, indices = np.unique(el_array.round(tol), axis=0, return_index=True)
    return np.array([el_array[index] for index in sorted(indices)])


class RandomQRotations(SphereGridNDim):

    algorithm_name = "randomQ"

    def _gen_grid_4D(self) -> NDArray:
        return random_quaternions(self.N)


class SystemERotations(SphereGridNDim):

    algorithm_name = "systemE"

    def _gen_grid_4D(self) -> NDArray:
        num_points = 1
        rot_matrices = []
        while len(rot_matrices) < self.N:
            phis = np.linspace(0, 2 * pi, num_points)
            thetas = np.linspace(0, 2 * pi, num_points)
            psis = np.linspace(0, 2 * pi, num_points)
            euler_meshgrid = np.array(np.meshgrid(*(phis, thetas, psis)), dtype=float)
            euler_meshgrid = euler_meshgrid.reshape((3, -1)).T
            rotations = Rotation.from_euler("ZYX", euler_meshgrid)
            # remove non-unique rotational matrices
            rotations = _select_unique_rotations(rotations)
            rot_matrices = rotations.as_matrix()
            num_points += 1
        # maybe just shuffle rather than order
        np.random.shuffle(rot_matrices)
        # convert to a grid
        rotations = Rotation.from_matrix(rot_matrices[:self.N])
        return rotations.as_quat()


class RandomERotations(SphereGridNDim):

    algorithm_name = "randomE"

    def _gen_grid_4D(self) -> NDArray:
        euler_angles = 2 * pi * np.random.random((self.N, 3))
        rotations = Rotation.from_euler("ZYX", euler_angles)
        return rotations.as_quat()


class ZeroRotations(SphereGridNDim):

    algorithm_name = "zero"

    def _gen_grid_4D(self):
        self.N = 1
        rot_matrix = np.eye(3)
        rot_matrix = rot_matrix[np.newaxis, :]
        self.rotations = Rotation.from_matrix(rot_matrix)
        return self.rotations.as_quat()


class Cube4DRotations(SphereGridNDim):

    algorithm_name = "cube4D"
    polytope = Cube4DPolytope()

    def _gen_grid_4D(self):
        while len(self.polytope.get_node_coordinates()) < self.N:
            self.polytope.divide_edges()
        return self.polytope.get_N_ordered_points(self.N)


class IcoAndCube3DRotations(SphereGridNDim):

    algorithm_name: str = None
    polytope: Polytope = None

    def get_z_projection_grid(self):
        "In this case, directly construct the 3D object"

    def _gen_grid_3D(self):
        while len(self.polytope.get_node_coordinates()) < self.N:
            self.polytope.divide_edges()
        return self.polytope.get_N_ordered_points(self.N)

    def _gen_grid_4D(self):
        grid_z_arr = self._gen_grid_3D()
        rotations = grid2rotation(grid_z_arr, grid_z_arr, grid_z_arr)
        return rotations.as_quat()
        # grid_z = SphereGrid3D(grid_z_arr, N=self.N, gen_alg=self.gen_algorithm)
        # z_vec = np.array([0, 0, 1])
        # matrices = np.zeros((desired_N, 3, 3))
        # for i in range(desired_N):
        #     matrices[i] = two_vectors2rot(z_vec, grid_z.get_grid()[i])
        # rot_z = Rotation.from_matrix(matrices)
        # grid_x_arr = rot_z.apply(np.array([1, 0, 0]))
        # grid_x = SphereGrid3D(grid_x_arr, N=desired_N, gen_alg=self.gen_algorithm)
        # grid_y_arr = rot_z.apply(np.array([0, 1, 0]))
        # grid_y = SphereGrid3D(grid_y_arr, N=desired_N, gen_alg=self.gen_algorithm)
        # self.from_grids(grid_x, grid_y, grid_z)


class IcoRotations(IcoAndCube3DRotations):

    algorithm_name = "ico"
    polytope = IcosahedronPolytope()


class Cube3DRotations(IcoAndCube3DRotations):

    algorithm_name = "cube3D"
    polytope = Cube3DPolytope()


class SphereGridFactory:

    @classmethod
    def create(cls, alg_name: str, N: int, dimensions: int, **kwargs) -> SphereGridNDim:
        if alg_name == "randomQ":
            selected_sub_obj = RandomQRotations(N=N, dimensions=dimensions, **kwargs)
        elif alg_name == "systemE":
            selected_sub_obj = SystemERotations(N=N, dimensions=dimensions, **kwargs)
        elif alg_name == "randomE":
            selected_sub_obj = RandomERotations(N=N, dimensions=dimensions, **kwargs)
        elif alg_name == "cube4D":
            selected_sub_obj = Cube4DRotations(N=N, dimensions=dimensions, **kwargs)
        elif alg_name == "zero":
            selected_sub_obj = ZeroRotations(N=N, dimensions=dimensions, **kwargs)
        elif alg_name == "ico":
            selected_sub_obj = IcoRotations(N=N, dimensions=dimensions, **kwargs)
        elif alg_name == "cube3D":
            selected_sub_obj = Cube3DRotations(N=N, dimensions=dimensions, **kwargs)
        else:
            raise ValueError(f"The algorithm {alg_name} not familiar to QuaternionGridFactory.")
        selected_sub_obj.gen_grid()
        return selected_sub_obj


class ConvergenceSphereGridFactory:

    def __init__(self, alg_name: str, dimensions: int, N_set = None, **kwargs):
        if N_set is None:
            N_set = SMALL_NS
        self.N_set = N_set
        self.alg_name = alg_name
        self.dimensions = dimensions
        self.list_sphere_grids = self.create(alg_name, dimensions, self.N_set, **kwargs)

    def get_name(self):
        return f"convergence_{self.alg_name}_{self.dimensions}d"

    def create(self, alg_name, dimensions, N_set, **kwargs) -> list:
        list_sphere_grids = []
        for N in N_set:
            sg = SphereGridFactory.create(alg_name, N, dimensions, **kwargs)
            list_sphere_grids.append(sg)
        return list_sphere_grids

    def get_spherical_voronoi_areas(self):
        data = []
        for N, sg in zip(self.N_set, self.list_sphere_grids):
            ideal_area = surface_per_cell_ideal(N, r=1)
            try:
                real_areas = sg.get_voronoi_areas()
            except ValueError:
                real_areas = [np.NaN] * sg.N
            for area in real_areas:
                data.append([N, ideal_area, area])
        df = pd.DataFrame(data, columns=["N", "ideal area", "sph. Voronoi cell area"])
        return df


