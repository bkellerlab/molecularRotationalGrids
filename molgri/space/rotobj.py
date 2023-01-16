from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from molgri.space.analysis import prepare_statistics, write_statistics
from molgri.space.utils import random_quaternions
from molgri.constants import UNIQUE_TOL, EXTENSION_GRID_FILES, GRID_ALGORITHMS, NAME2PRETTY_NAME
from molgri.molecules.parsers import GridNameParser
from molgri.paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_STAT
from molgri.space.polytopes import Cube4DPolytope, IcosahedronPolytope, Cube3DPolytope
from molgri.space.rotations import rotation2grid, grid2rotation, grid2quaternion, grid2euler, two_vectors2rot
from molgri.wrappers import time_method


class GridInfo:

    def __init__(self, point_array: NDArray, N: int, gen_alg: str = None):
        """
        Generate a grid with one of generation algorithms.

        Args:
            point_array: (N, 3) array in which each row is a 3D point with norm 1
            gen_alg: info with which alg the grid was created
        """
        assert gen_alg in GRID_ALGORITHMS, f"{gen_alg} is not a valid generation algorithm name"
        self.grid = point_array
        self.N = N
        self.standard_name = f"{gen_alg}_{N}"
        self.decorator_label = f"rotation grid {self.standard_name}"
        self.nn_dist_arch = None
        self.nn_dist_cup = None
        self.short_statistics_path = f"{PATH_OUTPUT_STAT}{self.standard_name}_short_stat.txt"
        self.statistics_path = f"{PATH_OUTPUT_STAT}{self.standard_name}_full_stat.csv"
        assert isinstance(self.grid, np.ndarray), "A grid must be a numpy array!"
        if len(point_array) > N:
            self._order_points()
        assert self.grid.shape == (N, 3), f"Grid not of correct shape! {self.grid.shape} instead of {(N, 3)}"
        assert np.allclose(np.linalg.norm(self.grid, axis=1), 1, atol=10**(-UNIQUE_TOL))

    def get_grid(self) -> np.ndarray:
        return self.grid

    def __len__(self):
        return self.N

    def save_grid(self, additional_name=""):
        if additional_name:
            additional_name = f"_{additional_name}"
        np.save(f"{PATH_OUTPUT_ROTGRIDS}{self.standard_name}{additional_name}.{EXTENSION_GRID_FILES}", self.get_grid())

    def save_grid_txt(self):
        # noinspection PyTypeChecker
        np.savetxt(f"{PATH_OUTPUT_ROTGRIDS}{self.standard_name}.txt", self.grid)

    def _order_points(self):
        """
        You are provided with a (possibly) unordered array of rotations saved in self.rotations. You must re-order
        it so that the coverage is maximised at every step.

        Additionally, truncate at self.N.
        """
        self.grid = order_elements(self.grid, self.N, start_i=1)

    def save_statistics(self, num_random: int = 100, print_message=False, alphas=None):
        stat_data, full_data = prepare_statistics(self.get_grid(), alphas, d=3, num_rand_points=num_random)
        write_statistics(stat_data, full_data, self.short_statistics_path, self.statistics_path,
                         num_random, name=self.standard_name, dimensions=3,
                         print_message=print_message)


class RotationsObject(ABC):

    def __init__(self, N: int = None, gen_algorithm: str = None, use_saved=True, time_generation=False):
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.rotations = None
        self.N = N
        self.gen_algorithm = gen_algorithm
        self.standard_name = f"{gen_algorithm}_{N}"
        self.decorator_label = f"{NAME2PRETTY_NAME[self.gen_algorithm]} with {self.N} points"
        self.short_statistics_path = f"{PATH_OUTPUT_STAT}{self.standard_name}_short_stat_rotobj.txt"
        self.statistics_path = f"{PATH_OUTPUT_STAT}{self.standard_name}_full_stat_rotobj.csv"
        if time_generation:
            gen_func = self.gen_and_time
        else:
            gen_func = self.gen_rotations
        if use_saved:
            try:
                grid_x_arr = np.load(f"{PATH_OUTPUT_ROTGRIDS}{gen_algorithm}_{N}_x.{EXTENSION_GRID_FILES}")
                grid_x = GridInfo(grid_x_arr, self.N, self.gen_algorithm)
                grid_y_arr = np.load(f"{PATH_OUTPUT_ROTGRIDS}{gen_algorithm}_{N}_y.{EXTENSION_GRID_FILES}")
                grid_y = GridInfo(grid_y_arr, self.N, self.gen_algorithm)
                grid_z_arr = np.load(f"{PATH_OUTPUT_ROTGRIDS}{gen_algorithm}_{N}_z.{EXTENSION_GRID_FILES}")
                grid_z = GridInfo(grid_z_arr, self.N, self.gen_algorithm)
                self.from_grids(grid_x, grid_y, grid_z)
            except FileNotFoundError:
                gen_func()
                self.save_all()
        else:
            gen_func()
            self.save_all()

    @abstractmethod
    def gen_rotations(self):
        pass

    @time_method
    def gen_and_time(self):
        self.gen_rotations()

    def get_grid_z_as_array(self) -> NDArray:
        return self.grid_z.get_grid()

    def get_grid_z_as_grid(self) -> GridInfo:
        return self.grid_z

    def _order_rotations(self):
        """
        You are provided with a (possibly) unordered array of rotations saved in self.rotations. You must re-order
        it so that the coverage is maximised at every step.

        Additionally, truncate at self.N.
        """
        rot_quaternions = self.rotations.as_quat()
        rot_quaternions = order_elements(rot_quaternions, self.N, start_i=1)
        self.rotations = Rotation.from_quat(rot_quaternions)
        self.from_rotations(self.rotations)

    def _select_unique_rotations(self):
        rot_matrices = self.rotations.as_matrix()
        rot_matrices = np.unique(np.round(rot_matrices, UNIQUE_TOL), axis=0)
        self.rotations = Rotation.from_matrix(rot_matrices)

    def from_rotations(self, rotations: Rotation):
        self.rotations = rotations
        grid_x, grid_y, grid_z = rotation2grid(rotations)
        self.grid_x = GridInfo(grid_x, N=len(rotations), gen_alg=self.gen_algorithm)
        self.grid_y = GridInfo(grid_y, N=len(rotations), gen_alg=self.gen_algorithm)
        self.grid_z = GridInfo(grid_z, N=len(rotations), gen_alg=self.gen_algorithm)
        self._determine_N()

    def from_grids(self, grid_x: GridInfo, grid_y: GridInfo, grid_z: GridInfo):
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
        stat_data, full_data = prepare_statistics(self.rotations.as_quat(), alphas, d=4, num_rand_points=num_random)
        write_statistics(stat_data, full_data, self.short_statistics_path, self.statistics_path,
                         num_random, name=self.standard_name, dimensions=4,
                         print_message=print_message)


def order_elements(points: np.ndarray, N: int, start_i: int = 1) -> NDArray:
    """
    You are provided with a (possibly) unordered grid and return a grid with N points ordered in such a way that
    these N points have the best possible coverage.

    Args:
        points: grid, array of shape (L, 3) to be ordered where L should be >= N
        N: number of grid points wished at the end
        start_i: from which index to start ordering (in case the first i elements already ordered)

    Returns:
        an array of shape (N, 3) ordered in such a way that these N points have the best possible coverage.
    """

    def select_next_el(quaternion_list, i):
        distances = cdist(quaternion_list[i:], quaternion_list[:i], metric="cosine")
        distances.sort()
        nn_dist = distances[:, 0]
        index_max = np.argmax(nn_dist)
        quaternion_list[[i, i + index_max]] = quaternion_list[[i + index_max, i]]
        return quaternion_list

    if N > len(points):
        raise ValueError(f"N>len(grid)! Only {len(points)} points can be returned!")
    for index in range(start_i, min(len(points), N)):
        points = select_next_el(points, index)
    return points[:N]


class RandomQRotations(RotationsObject):

    def gen_rotations(self):
        assert self.N is not None, "Select the number of points N!"
        quaternions = random_quaternions(self.N)
        self.from_rotations(Rotation.from_quat(quaternions))


class SystemERotations(RotationsObject):

    def gen_rotations(self):
        assert self.N is not None, "Select the number of points N!"
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
        self._order_rotations()
        # convert to a grid
        self.from_rotations(self.rotations)
        # self.grid = euler2grid(euler_meshgrid)


class RandomERotations(RotationsObject):

    def gen_rotations(self):
        euler_angles = 2 * pi * np.random.random((self.N, 3))
        self.from_rotations(Rotation.from_euler("ZYX", euler_angles))


class ZeroRotations(RotationsObject):

    def __init__(self, N: int = None, gen_algorithm: str = None, use_saved=True, time_generation=False):
        assert N == 1, "Zero grid only makes sense for N=1"
        super().__init__(N=N, gen_algorithm=gen_algorithm, use_saved=use_saved, time_generation=time_generation)

    def gen_rotations(self):
        self.N = 1
        rot_matrix = np.eye(3)
        rot_matrix = rot_matrix[np.newaxis, :]
        self.rotations = Rotation.from_matrix(rot_matrix)
        self.from_rotations(self.rotations)


class PolyhedronRotations(RotationsObject):

    def __init__(self, polyhedron, *args, **kwargs):
        self.polyhedron = polyhedron()
        super().__init__(*args, **kwargs)

    def gen_rotations(self):
        while self.polyhedron.G.number_of_nodes() < self.N:
            self.polyhedron.divide_edges()


class Cube4DRotations(PolyhedronRotations):

    def __init__(self, N: int, gen_algorithm, **kwargs):
        super().__init__(polyhedron=Cube4DPolytope, N=N, gen_algorithm=gen_algorithm, **kwargs)

    def gen_rotations(self):
        super().gen_rotations()
        rotations = np.array([y["projection"] for x, y in self.polyhedron.G.nodes(data=True)]).squeeze()
        self.rotations = Rotation.from_quat(rotations)
        self._order_rotations()
        self.from_rotations(self.rotations)


class IcoAndCube3DRotations(PolyhedronRotations):

    def gen_rotations(self):
        desired_N = self.N
        super().gen_rotations()
        grid_z_arr = np.array([y["projection"] for x, y in self.polyhedron.G.nodes(data=True)]).squeeze()
        # TODO: should you try ordering grid points?
        grid_z = GridInfo(grid_z_arr, N=self.N, gen_alg=self.gen_algorithm)

        z_vec = np.array([0, 0, 1])
        matrices = np.zeros((desired_N, 3, 3))
        for i in range(desired_N):
            matrices[i] = two_vectors2rot(z_vec, grid_z.get_grid()[i])
        rot_z = Rotation.from_matrix(matrices)

        all_x_vec = np.zeros((desired_N, 3))
        all_y_vec = np.zeros((desired_N, 3))
        all_z_vec = np.zeros((desired_N, 3))
        all_x_vec[:, 0] = 1
        all_y_vec[:, 1] = 1
        all_z_vec[:, 2] = 1
        grid_x_arr = rot_z.apply(all_x_vec)            #R_xz.apply(grid_z.get_grid())
        grid_x = GridInfo(grid_x_arr, N=desired_N, gen_alg=self.gen_algorithm)
        grid_y_arr = rot_z.apply(all_y_vec)             #R_yz.apply(grid_z.get_grid(), inverse=True)
        grid_y = GridInfo(grid_y_arr, N=desired_N, gen_alg=self.gen_algorithm)

        # angles
        # from molgri.space.utils import angle_between_vectors
        # for vec1, vec2, vec3 in zip(grid_x.get_grid(), grid_y.get_grid(), grid_z.get_grid()):
        #     print("angle", angle_between_vectors(vec1, vec3)/pi, angle_between_vectors(vec2, vec3)/pi, angle_between_vectors(vec1, vec2)/pi)
        self.from_grids(grid_x, grid_y, grid_z)
        self.save_all()


# this is only a test, not sure if right
class IcoRotations(IcoAndCube3DRotations):

    def __init__(self, N: int, gen_algorithm, **kwargs):
        super().__init__(polyhedron=IcosahedronPolytope, N=N, gen_algorithm=gen_algorithm, **kwargs)


class Cube3DRotations(IcoAndCube3DRotations):

    def __init__(self, N: int, gen_algorithm, **kwargs):
        super().__init__(polyhedron=Cube3DPolytope, N=N, gen_algorithm=gen_algorithm, **kwargs)


def build_rotations_from_name(grid_name: str, b_or_o="o", use_saved=True, **kwargs) -> RotationsObject:
    gnp = GridNameParser(grid_name, b_or_o)
    return build_rotations(gnp.N, gnp.algo, use_saved=use_saved, **kwargs)


def build_rotations(N: int, algo: str, **kwargs) -> RotationsObject:
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
    rot_obj = name2rotation[algo](N, gen_algorithm=algo, **kwargs)
    return rot_obj


def build_grid_from_name(grid_name: str, b_or_o="o", use_saved=True, **kwargs) -> GridInfo:
    return build_rotations_from_name(grid_name, b_or_o=b_or_o, use_saved=use_saved, **kwargs).get_grid_z_as_grid()


def build_grid(N: int, algo: str, **kwargs) -> GridInfo:
    return build_rotations(N=N, algo=algo, **kwargs).get_grid_z_as_grid()


if __name__ == "__main__":
    for alg in GRID_ALGORITHMS[:-1]:
        print(alg)
        rots = build_rotations_from_name(f"{alg}_{72}", use_saved=False)
        print(len(rots.rotations), len(rots.get_grid_z_as_array()))
