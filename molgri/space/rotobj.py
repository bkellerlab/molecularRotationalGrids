import os
from abc import ABC, abstractmethod
from typing import List, Type

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial.transform import Rotation

from molgri.space.analysis import prepare_statistics, write_statistics
from molgri.space.utils import random_quaternions, standardise_quaternion_set
from molgri.constants import UNIQUE_TOL, EXTENSION_GRID_FILES, NAME2PRETTY_NAME
from molgri.molecules.parsers import GridNameParser
from molgri.paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_STAT
from molgri.space.polytopes import Cube4DPolytope, IcosahedronPolytope, Cube3DPolytope
from molgri.space.rotations import rotation2grid, grid2rotation, grid2quaternion, grid2euler, two_vectors2rot, \
    rotation2grid4vector
from molgri.wrappers import time_method


class SphereGridNDim(ABC):

    def __init__(self, dimensions: int, N: int = None, gen_alg: str = None, use_saved: bool = True,
                 print_messages: bool = True, time_generation: bool = False):
        self.dimensions = dimensions
        self.N = N
        self.gen_algorithm = gen_alg
        self.use_saved = use_saved
        self.time_generation = time_generation
        self.print_messages = print_messages
        self.grid: NDArray = None

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
        assert isinstance(self.grid, np.ndarray), "A grid must be a numpy array!"
        assert self.grid.shape == (self.N, self.dimensions), f"Grid not of correct shape!"
        assert np.allclose(np.linalg.norm(self.grid, axis=1), 1, atol=10 ** (-UNIQUE_TOL)), "A grid must have norm 1!"

    @abstractmethod
    def _gen_grid(self) -> NDArray:
        pass

    @time_method
    def gen_and_time(self) -> NDArray:
        return self._gen_grid()

    ##################################################################################################################
    #                      name and path getters
    ##################################################################################################################

    def get_standard_name(self, with_dim=False) -> str:
        output = self.gen_algorithm
        if self.N is not None:
            output += f"_{self.N}"
        if with_dim:
            output += f"_{self.dimensions}d"
        return output

    def get_decorator_name(self) -> str:
        return f"{NAME2PRETTY_NAME[self.gen_algorithm]} algorithm, {self.N} points"

    def get_grid_path(self, extension=EXTENSION_GRID_FILES) -> str:
        return f"{PATH_OUTPUT_ROTGRIDS}{self.get_standard_name(with_dim=True)}.{extension}"

    def get_statistics_path(self, extension) -> str:
        return f"{PATH_OUTPUT_STAT}{self.get_standard_name(with_dim=True)}.{extension}"

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
                         num_random, name=self.get_standard_name(), dimensions=self.dimensions,
                         print_message=self.print_messages)

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


class SphereGrid3D(SphereGridNDim):

    def __init__(self, point_array: NDArray, N: int, gen_alg: str = None, **kwargs):
        """
        A grid consisting of points on a sphere in 3D.

        Args:
            point_array: (N, 3) array in which each row is a 3D point with norm 1
            gen_alg: info with which alg the grid was created
        """
        super().__init__(dimensions=3, N=N, gen_alg=gen_alg, **kwargs)
        self.grid = point_array

    def _gen_grid(self):
        pass


class SphereGrid4D(SphereGridNDim, ABC):

    # property of the sub-class is the algorithm name
    algorithm_name = "None"

    def __init__(self, N: int = None, **kwargs):
        """
        When initiating a class, start by saving required properties but not yet generating any points. For actually
        creating/reading points from a file, you need to run the gen_rotations(N) method.
        Args:
            N:
            gen_algorithm:
            use_saved:
            time_generation:
            print_warnings:
        """
        super().__init__(N=N, dimensions=4, gen_alg=self.algorithm_name, **kwargs)
        self.rotations = None

    def _set_N(self, N: int):
        assert N > 0, "N must be a positive integer"
        self.N = N

    def create_rotations(self, N: int):
        """
        Standard method to run in order to create/use saved rotational quaternions

        Args:
            N: number of points
        """
        self._set_N(N)
        gen_func = self.gen_and_time if self.time_generation else self.gen_rotations

        if self.use_saved:
            try:
                grid_x_arr = np.load(f"{PATH_OUTPUT_ROTGRIDS}{self.get_standard_name()}_x.{EXTENSION_GRID_FILES}")
                grid_x = SphereGrid3D(grid_x_arr, self.N, self.gen_algorithm)
                grid_y_arr = np.load(f"{PATH_OUTPUT_ROTGRIDS}{self.get_standard_name()}_y.{EXTENSION_GRID_FILES}")
                grid_y = SphereGrid3D(grid_y_arr, self.N, self.gen_algorithm)
                grid_z_arr = np.load(f"{PATH_OUTPUT_ROTGRIDS}{self.get_standard_name()}_z.{EXTENSION_GRID_FILES}")
                grid_z = SphereGrid3D(grid_z_arr, self.N, self.gen_algorithm)
                self.from_grids(grid_x, grid_y, grid_z)
                return
            except FileNotFoundError:
                gen_func()
        else:
            gen_func()
            self.save_all()
        # grid of correct shape
        z_array = self.get_grid_z_as_array()
        assert z_array.shape == (self.N, 3), f"Wrong shape: {z_array.shape} != {(self.N, 3)}"
        # rotations unique
        quats = self.rotations.as_quat()
        if self.print_warnings and len(quats) != len(provide_unique(quats, as_quat=True)):
            print(f"Warning! {len(quats) - len(provide_unique(quats, as_quat=True))} "
                  f"quaternions of {self.get_standard_name()} non-unique (distance < 10^-{UNIQUE_TOL}).")
        # grids unique
        if self.print_warnings and len(z_array) != len(provide_unique(z_array)):
            print(f"Warning! {len(z_array) - len(provide_unique(z_array))} grid points"
                  f" of {self.get_standard_name()} non-unique (distance < 10^-{UNIQUE_TOL}).")

    # def get_grid_z_as_array(self) -> NDArray:
    #     return self.grid_z.get_grid()

    def get_z_projection_grid(self) -> SphereGrid3D:
        if self.rotations is None:
            self.rotations = Rotation.from_quat(self.get_grid_as_array())
        array_3D = rotation2grid4vector(self.rotations)
        # here use default projection, rewrite for Ico and Cube3D
        return SphereGrid3D(array_3D, N=self.N, gen_alg=self.gen_algorithm, time_generation=self.time_generation,
                            use_saved=self.use_saved, print_messages=self.print_messages)

    def _select_unique_rotations(self):
        rot_matrices = self.rotations.as_matrix()
        rot_matrices = np.unique(np.round(rot_matrices, UNIQUE_TOL), axis=0)
        self.rotations = Rotation.from_matrix(rot_matrices)

    def from_rotations(self, rotations: Rotation):
        self.rotations = rotations
        grid_x, grid_y, grid_z = rotation2grid(rotations)
        self.grid_x = SphereGrid3D(grid_x, N=len(rotations), gen_alg=self.gen_algorithm)
        self.grid_y = SphereGrid3D(grid_y, N=len(rotations), gen_alg=self.gen_algorithm)
        self.grid_z = SphereGrid3D(grid_z, N=len(rotations), gen_alg=self.gen_algorithm)
        self._determine_N()

    def from_grids(self, grid_x: SphereGrid3D, grid_y: SphereGrid3D, grid_z: SphereGrid3D):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.rotations = self.as_rotation_object()
        self._determine_N()

    def _determine_N(self):
        if self.rotations is None:
            self.N = None
        else:
            self.N = len(self.rotations)

    def as_rotation_object(self) -> Rotation:
        return grid2rotation(self.grid_x.get_grid(), self.grid_y.get_grid(), self.grid_z.get_grid())

    def get_old_rotation_objects(self):
        z_vec = np.array([0, 0, 1])
        matrices = np.zeros((self.N, 3, 3))
        for i in range(self.N):
            matrices[i] = two_vectors2rot(z_vec, self.grid_z.get_grid()[i])
        return Rotation.from_matrix(matrices)

    def as_quaternion(self) -> NDArray:
        return grid2quaternion(self.grid_x.get_grid(), self.grid_y.get_grid(), self.grid_z.get_grid())

    def as_euler(self) -> NDArray:
        return grid2euler(self.grid_x.get_grid(), self.grid_y.get_grid(), self.grid_z.get_grid())

    def save_all(self):
        subgrids = (self.grid_x, self.grid_y, self.grid_z)
        labels = ("x", "y", "z")
        for label, sub_grid in zip(labels, subgrids):
            sub_grid.save_grid(additional_name=label)

    def save_statistics(self, num_random: int = 100, print_message=False, alphas=None):
        short_statistics_path = f"{PATH_OUTPUT_STAT}{self.get_standard_name()}_short_stat_rotobj.txt"
        statistics_path = f"{PATH_OUTPUT_STAT}{self.get_standard_name()}_full_stat_rotobj.csv"
        # self.get_old_rotation_objects().as_quat()
        stat_data, full_data = prepare_statistics(self.rotations.as_quat(), alphas, d=4, num_rand_points=num_random)
        write_statistics(stat_data, full_data, short_statistics_path, statistics_path,
                         num_random, name=self.get_standard_name(), dimensions=4,
                         print_message=print_message)


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


class RandomQRotations(SphereGrid4D):

    algorithm_name = "randomQ"

    def _gen_grid(self) -> NDArray:
        assert self.N is not None, "Select the number of points N!"
        return random_quaternions(self.N)


class SystemERotations(SphereGrid4D):

    algorithm_name = "systemE"

    def _gen_grid(self) -> NDArray:
        num_points = 1
        rot_matrices = []
        while len(rot_matrices) < self.N:
            phis = np.linspace(0, 2 * pi, num_points)
            thetas = np.linspace(0, 2 * pi, num_points)
            psis = np.linspace(0, 2 * pi, num_points)
            euler_meshgrid = np.array(np.meshgrid(*(phis, thetas, psis)), dtype=float)
            euler_meshgrid = euler_meshgrid.reshape((3, -1)).T
            self.rotations = Rotation.from_euler("ZYX", euler_meshgrid)
            # remove non-unique rotational matrices
            self._select_unique_rotations()
            rot_matrices = self.rotations.as_matrix()
            num_points += 1
        # maybe just shuffle rather than order
        np.random.shuffle(rot_matrices)
        # convert to a grid
        self.rotations = Rotation.from_matrix(rot_matrices[:self.N])
        return self.rotations.as_quat()


class RandomERotations(SphereGrid4D):

    algorithm_name = "randomE"

    def _gen_grid(self) -> NDArray:
        euler_angles = 2 * pi * np.random.random((self.N, 3))
        self.rotations = Rotation.from_euler("ZYX", euler_angles)
        return self.rotations.as_quat()


class ZeroRotations(SphereGrid4D):

    algorithm_name = "zero"

    def _gen_grid(self):
        self.N = 1
        rot_matrix = np.eye(3)
        rot_matrix = rot_matrix[np.newaxis, :]
        self.rotations = Rotation.from_matrix(rot_matrix)
        return self.rotations.as_quat()


class PolyhedronRotations(SphereGrid4D):

    def __init__(self, polyhedron, *args, **kwargs):
        self.polyhedron = polyhedron()
        super().__init__(*args, **kwargs)

    def _gen_grid(self):
        while len(self.polyhedron.get_node_coordinates()) < self.N:
            self.polyhedron.divide_edges()


class Cube4DRotations(PolyhedronRotations):

    algorithm_name = "cube4D"

    def __init__(self, **kwargs):
        super().__init__(polyhedron=Cube4DPolytope, **kwargs)

    def _gen_grid(self):
        super()._gen_grid()
        rotations = self.polyhedron.get_N_ordered_points(self.N)
        self.rotations = Rotation.from_quat(rotations)
        self.from_rotations(self.rotations)


class IcoAndCube3DRotations(PolyhedronRotations):

    def get_z_projection_grid(self):
        "In this case, directly construct the 3D object"

    def _gen_grid(self):
        desired_N = self.N
        super()._gen_grid()
        grid_z_arr = self.polyhedron.get_N_ordered_points(desired_N)
        grid_z = SphereGrid3D(grid_z_arr, N=self.N, gen_alg=self.gen_algorithm)
        z_vec = np.array([0, 0, 1])
        matrices = np.zeros((desired_N, 3, 3))
        for i in range(desired_N):
            matrices[i] = two_vectors2rot(z_vec, grid_z.get_grid()[i])
        rot_z = Rotation.from_matrix(matrices)
        grid_x_arr = rot_z.apply(np.array([1, 0, 0]))
        grid_x = SphereGrid3D(grid_x_arr, N=desired_N, gen_alg=self.gen_algorithm)
        grid_y_arr = rot_z.apply(np.array([0, 1, 0]))
        grid_y = SphereGrid3D(grid_y_arr, N=desired_N, gen_alg=self.gen_algorithm)
        self.from_grids(grid_x, grid_y, grid_z)
        self.save_all()


class IcoRotations(IcoAndCube3DRotations):

    algorithm_name = "ico"

    def __init__(self, **kwargs):
        super().__init__(polyhedron=IcosahedronPolytope, **kwargs)


class Cube3DRotations(IcoAndCube3DRotations):

    algorithm_name = "cube3D"

    def __init__(self, **kwargs):
        super().__init__(polyhedron=Cube3DPolytope, **kwargs)


class AllRotObjObjects:
    sub_obj = List[Type[SphereGridNDim]]



def build_rotations_from_name(grid_name: str, b_or_o="o", use_saved=False, **kwargs) -> SphereGrid4D:
    gnp = GridNameParser(grid_name, b_or_o)
    return build_rotations(gnp.N, gnp.algo, use_saved=use_saved, **kwargs)


def build_rotations(N: int, algo: str, use_saved=False, **kwargs) -> SphereGrid4D:
    name2rotation = {"randomQ": RandomQRotations,
                     "systemE": SystemERotations,
                     "randomE": RandomERotations,
                     "cube4D": Cube4DRotations,
                     "zero": ZeroRotations,
                     "ico": IcoRotations,
                     "cube3D": Cube3DRotations
                     }
    if algo not in name2rotation.keys():
        raise ValueError(f"Algorithm {algo} is not a valid grid type. "
                         f"Try 'ico', 'cube3D' ...")
    assert isinstance(N, int), f"Number of grid points must be an integer, currently N={N}"
    assert N >= 0, f"Number of grid points cannot be negative, currently N={N}"
    rot_obj = name2rotation[algo](N, gen_algorithm=algo, use_saved=use_saved, **kwargs)
    return rot_obj


def build_grid_from_name(grid_name: str, b_or_o="o", use_saved=False, **kwargs) -> SphereGrid3D:
    return build_rotations_from_name(grid_name, b_or_o=b_or_o, use_saved=use_saved, **kwargs).get_z_projection_grid()


def build_grid(N: int, algo: str, use_saved=False, **kwargs) -> SphereGrid3D:
    return build_rotations(N=N, algo=algo, use_saved=use_saved, **kwargs).get_z_projection_grid()


class SphereGridFactory:

    @classmethod
    def create(cls, alg_name: str, N: int, dimensions: int, **kwargs) -> SphereGridNDim:
        if alg_name == "randomQ":
            selected_sub_obj = RandomQRotations(N=N, **kwargs)
        elif alg_name == "systemE":
            selected_sub_obj = SystemERotations(N=N, **kwargs)
        elif alg_name == "randomE":
            selected_sub_obj = RandomERotations(N=N,  **kwargs)
        elif alg_name == "cube4D":
            selected_sub_obj = Cube4DRotations(N=N, **kwargs)
        elif alg_name == "zero":
            selected_sub_obj = ZeroRotations(N=N,  **kwargs)
        elif alg_name == "ico":
            selected_sub_obj = IcoRotations(N=N, **kwargs)
        elif alg_name == "cube3D":
            selected_sub_obj = Cube3DRotations(N=N, **kwargs)
        else:
            raise ValueError(f"The algorithm {alg_name} not familiar to QuaternionGridFactory.")
        selected_sub_obj.gen_grid()
        if dimensions == 3:
            return selected_sub_obj.get_z_projection_grid()
        else:
            return selected_sub_obj