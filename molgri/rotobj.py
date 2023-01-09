from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from molgri.analysis import random_quaternions
from molgri.constants import UNIQUE_TOL
from molgri.grids import Grid, project_grid_on_sphere
from molgri.parsers import GridNameParser
from molgri.rotations import rotation2grid, grid2rotation, grid2quaternion, grid2euler


class RotationsObject(ABC):

    def __init__(self, N: int = None, gen_algorithm: str = None):
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.rotations = None
        self.N = N
        self.gen_algorithm = gen_algorithm
        self.standard_name = f"{gen_algorithm}_{N}"
        self.gen_rotations(self.N, self.gen_algorithm)

    @abstractmethod
    def gen_rotations(self, N: int = None, gen_algorithm: str = None):
        pass

    def _order_rotations(self):
        """
        You are provided with a (possibly) unordered array of rotations saved in self.rotations. You must re-order
        it so that the coverage is maximised at every step.

        Additionally, truncate at self.N.
        """
        rot_quaternions = self.rotations.as_quat()
        if self.N > len(rot_quaternions):
            raise ValueError(f"N>len(grid)! Only {len(rot_quaternions)} points can be returned!")
        for index in range(1, self.N):
            rot_quaternions = select_next_rotation(rot_quaternions, index)
        rot_quaternions = rot_quaternions[:self.N]
        self.rotations = Rotation.from_quat(rot_quaternions)

    def _select_unique_rotations(self):
        rot_matrices = self.rotations.as_matrix()
        rot_matrices = np.unique(np.round(rot_matrices, UNIQUE_TOL), axis=0)
        self.rotations = Rotation.from_matrix(rot_matrices)

    def from_rotations(self, rotations: Rotation):
        self.rotations = rotations
        self.grid_x, self.grid_y, self.grid_z = rotation2grid(rotations)
        self._determine_N()

    def from_grids(self, grid_x: Grid, grid_y: Grid, grid_z: Grid):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.rotations = self.as_rotation_object()
        self._determine_N()

    def _determine_N(self):
        if self.grid_x is None and self.grid_y is None and self.grid_z is None:
            self.N = None
        else:
            assert len(self.grid_x) == len(self.grid_y) == len(self.grid_z)
            self.N = len(self.grid_x)

    def as_rotation_object(self) -> Rotation:
        return grid2rotation(self.grid_x, self.grid_y, self.grid_z)

    def as_quaternion(self) -> NDArray:
        return grid2quaternion(self.grid_x, self.grid_y, self.grid_z)

    def as_euler(self) -> NDArray:
        return grid2euler(self.grid_x, self.grid_y, self.grid_z)

    def save_all(self):
        subgrids = (self.grid_x, self.grid_y, self.grid_z)
        labels = ("x", "y", "z")
        for label, sub_grid in zip(labels, subgrids):
            if not sub_grid.standard_name.endswith(f"_{label}"):
                sub_grid.standard_name = f"{sub_grid.standard_name}_{label}"
                sub_grid.save_grid()


def select_next_rotation(quaternion_list, i):
    """
    Provide an array of quaternions where the first i are already sorted. Find the best next quaternion out of points
    in quaternion_list[i:] to maximise coverage

    Args:
        quaternion_list: array of shape (L, 4) where elements up to i are already ordered
        i: index how far the array is already ordered (up to bun not including i).

    Returns:
        set_grid_points where the ith element in swapped with the best possible next grid point
    """
    distances = cdist(quaternion_list[i:], quaternion_list[:i], metric="cosine") # TODO: is this right?
    distances.sort()
    nn_dist = distances[:, 0]
    index_max = np.argmax(nn_dist)
    quaternion_list[[i, i + index_max]] = quaternion_list[[i + index_max, i]]
    return quaternion_list


class RandomQRotations(RotationsObject):

    def gen_rotations(self, N: int = None, gen_algorithm="randomQ"):
        assert N is not None, "Select the number of points N!"
        quaternions = random_quaternions(N)
        self.from_rotations(Rotation.from_quat(quaternions))


class SystemERotations(RotationsObject):

    def gen_rotations(self, N: int = None, gen_algorithm="randomQ"):
        assert N is not None, "Select the number of points N!"
        num_points = 1
        rot_matrices = []
        while len(rot_matrices) < N:
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

    def gen_rotations(self, N: int = None, gen_algorithm="randomE"):
        euler_angles = 2 * pi * np.random.random((N, 3))
        self.from_rotations(Rotation.from_euler("ZYX", euler_angles))


class Cube4DRotations(RotationsObject):

    def __init__(self, *args, **kwargs):
        self.d = 4
        super().__init__(*args, **kwargs)

    def gen_rotations(self, N: int = None, gen_algorithm: str = None):
        self.N = N
        rot_matrices = []
        num_divisions = 1
        state_before = np.random.get_state()
        while len(rot_matrices) < self.N:
            grid_qua = self._full_d_dim_grid()
            grid_qua = self._select_only_faces(grid_qua)
            grid_qua = project_grid_on_sphere(grid_qua)
            # select only half the sphere
            grid_qua = grid_qua[grid_qua[:, self.d - 1] >= 0, :]
            self.rotations = Rotation.from_quat(grid_qua)
            # remove non-unique rotational matrices
            self._select_unique_rotations()
            rot_matrices = self.rotations.as_matrix()
            num_divisions += 1
        np.random.set_state(state_before)
        self._order_rotations()

    def _full_d_dim_grid(self, dtype=np.float64) -> np.ndarray:
        """
        This is a function to create a classical grid of a d-dimensional cube. It creates a grid over the entire
        (hyper)volume of the (hyper)cube.

        This is a unit cube between -sqrt(1/d) and sqrt(1/d) in all dimensions where d = num of dimensions.

        Args:
            dtype: forwarded to linspace while creating a grid

        Returns:
            a meshgrid of dimension (d, n, n, .... n) where n is repeated d times
        """
        side = np.linspace(-1, 1, self.N, dtype=dtype)
        # repeat the same n points d times and then make a new line of the array every d elements
        sides = np.tile(side, self.d)
        sides = sides[np.newaxis, :].reshape((self.d, self.N))
        # create a grid by meshing every line of the sides array
        return np.array(np.meshgrid(*sides))

    def _select_only_faces(self, grid: np.ndarray):
        """
        Take a meshgrid (d, n, n, ... n)  and return an array of points (N, d) including only the points that
        lie on the faces of the grid, so the edge points in at least one of dimensions.

        Args:
            grid: numpy array (d, n, n, ... n) containing grid points

        Returns:
            points (N, d) where N is the number of edge points and d the dimension
        """
        assert self.d == len(grid)
        set_grids = []
        for swap_i in range(self.d):
            meshgrid_swapped = np.swapaxes(grid, axis1=1, axis2=(1 + swap_i))
            set_grids.append(meshgrid_swapped[:, 0, ...])
            set_grids.append(meshgrid_swapped[:, -1, ...])

        result = np.hstack(set_grids).reshape((self.d, -1)).T
        return np.unique(result, axis=0)


class ZeroRotations(RotationsObject):

    def gen_rotations(self, N=1, gen_algorithm="zero"):
        self.N = 1
        rot_matrix = np.eye(3)
        self.rotations = Rotation.from_matrix(rot_matrix)
        self.from_rotations(self.rotations)


def build_rotations_from_name(grid_name: str, **kwargs) -> RotationsObject:
    gnp = GridNameParser(grid_name)
    return build_rotations(gnp.N, gnp.algo, **kwargs)


def build_rotations(N: int, algo: str, **kwargs) -> RotationsObject:
    name2rotation = {"randomQ": RandomQRotations,
                     "systemE": SystemERotations,
                     "randomE": RandomERotations,
                     "cube4D": Cube4DRotations,
                     "zero": ZeroRotations
                     }
                 #
                 # "cube4D": Cube4DGrid,
                 # "systemE": SystemEGrid,
                 # "cube3D": Cube3DGrid,
                 # "ico": IcoGrid,
                 # "zero": ZeroGrid}
    if algo not in name2rotation.keys():
        raise ValueError(f"Algorithm {algo} is not a valid grid type. "
                         f"Try 'ico', 'cube3D' ...")
    assert isinstance(N, int), f"Number of grid points must be an integer, currently N={N}"
    assert N >= 0, f"Number of grid points cannot be negative, currently N={N}"
    rot_obj = name2rotation[algo](N, gen_algorithm=algo, **kwargs)
    return rot_obj


