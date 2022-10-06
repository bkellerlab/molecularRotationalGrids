from datetime import timedelta
from time import time
import numpy as np
from abc import ABC, abstractmethod
from scipy.constants import pi
from scipy.spatial.distance import cdist

from molgri.analysis.uniformity_measure import random_sphere_points
from molgri.grids.cube_grid import cube_grid_on_sphere, select_half_sphere
from molgri.grids.grid2rotation2grid import grid2euler, euler2grid, quaternion2grid, grid2quaternion
from molgri.grids.polytopes import CubePolytope, IcosahedronPolytope
from molgri.parsers.name_parser import NameParser
from molgri.grids.order_grid import order_grid_points
from molgri.my_constants import *
from molgri.paths import PATH_OUTPUT_ROTGRIDS


class Grid(ABC):

    def __init__(self, N: int, *, ordered: bool = True, use_saved: bool = False, gen_alg: str = None):
        """
        Generate a grid with one of generation algorithms.

        Args:
            gen_alg: MUST BE SET IN SUBCLASSES, algorithm name, see names given in SIX_METHOD_NAMES
            N: number of grid points
            ordered: if True order and truncate, else only truncate to N points
            use_saved: if True use saved grids if available
        """
        self.rn_gen = np.random.default_rng(DEFAULT_SEED)
        np.random.seed(DEFAULT_SEED)
        if gen_alg not in SIX_METHOD_NAMES:
            raise ValueError(f"{gen_alg} is not a valid generation algorithm name. Try 'ico', 'cube3D' ...")
        self.ordered = ordered
        self.N = N
        name_properties = {"grid_type": gen_alg, "num_grid_points": N, "ordering": ordered}
        self.standard_name = NameParser(name_properties).get_standard_name()
        self.grid = None
        self.time = 0
        self.nn_dist_arch = None
        self.nn_dist_cup = None
        # if this option enabled, search first if this grid has already been saved
        if use_saved:
            try:
                self.grid = np.load(f"{PATH_OUTPUT_ROTGRIDS}{self.standard_name}.npy")
            except FileNotFoundError:
                self.generate_and_time()
        else:
            self.generate_and_time()
        assert isinstance(self.grid, np.ndarray), "A grid must be a numpy array!"
        assert self.grid.shape == (N, 3), f"Grid not of correct shape! {self.grid.shape} instead of {(N, 3)}"
        assert np.allclose(np.linalg.norm(self.grid, axis=1), 1, atol=10**(-UNIQUE_TOL))

    def get_grid(self) -> np.ndarray:
        return self.grid

    def reduce_N(self, reduce_by=1):
        if reduce_by <= 0:
            return
        self.grid = self.grid[:-reduce_by]
        self.N = self.N - reduce_by
        self.time = 0
        self.nn_dist_arch = None
        self.nn_dist_cup = None

    def get_nn_distances(self, num_random: int = 500, unit="arch") -> np.ndarray:
        if unit == "arch" and self.nn_dist_arch is not None:
            return self.nn_dist_arch
        elif unit == "spherical_cap" and self.nn_dist_cup is not None:
            # noinspection PyTypeChecker
            return self.nn_dist_cup
        else:
            # nn_dist = random_axes_count_points(grid=self, alpha=pi/4, num_random_points=100)
            random_points = random_sphere_points(num_random)
            distances = cdist(random_points, self.grid, metric="cosine")
            distances.sort()
            nn_dist = distances[:, 0]
            if unit == "arch":
                self.nn_dist_arch = nn_dist
                return self.nn_dist_arch
            elif unit == "spherical_cap":
                self.nn_dist_cup = 2 * pi * (1 - np.cos(nn_dist))
                return self.nn_dist_cup
            else:
                raise ValueError(f"{unit} not a valid uniformity unit, try 'arch' or 'spherical_cup")

    def get_nn_overview(self, num_random: int = 1000, unit="arch"):
        nn_dist = self.get_nn_distances(num_random=num_random, unit=unit)
        min_dist = np.min(nn_dist)
        max_dist = np.max(nn_dist)
        average_dist = np.average(nn_dist)
        sd = np.std(nn_dist)
        return np.array([min_dist, max_dist, average_dist, sd])

    @abstractmethod
    def generate_grid(self):
        # order or truncate
        if self.ordered:
            self._order()
        else:
            self.grid = self.grid[:self.N]

    def generate_and_time(self, print_message=False):
        t1 = time()
        self.generate_grid()
        t2 = time()
        if print_message:
            print(f"Timing the generation of the grid {self.standard_name}: ", end="")
            print(f"{timedelta(seconds=t2-t1)} hours:minutes:seconds")
        self.time = t2 - t1

    def _order(self):
        self.grid = order_grid_points(self.grid, self.N)

    def as_quaternion(self) -> np.ndarray:
        quaternion_seq = grid2quaternion(self.grid)
        assert isinstance(quaternion_seq, np.ndarray), "A quaternion sequence must be a numpy array!"
        assert quaternion_seq.shape == (self.N, 4), f"Quaternion sequence not of correct shape!\
                                {quaternion_seq.shape} instead of {(self.N, 4)}"
        return quaternion_seq

    def as_euler(self) -> np.ndarray:
        euler_seq = grid2euler(self.grid)
        assert isinstance(euler_seq, np.ndarray), "An Euler sequence must be a numpy array!"
        assert euler_seq.shape == (self.N, 3), f"An Euler sequence not of correct shape!\
                                {euler_seq.shape} instead of {(self.N, 33)}"
        return euler_seq

    def save_grid(self):
        np.save(f"{PATH_OUTPUT_ROTGRIDS}{self.standard_name}.npy", self.grid)

    def save_violin_data(self, num_random: int = 1000, unit="arch"):
        violin_data = self.get_nn_distances(num_random, unit=unit)
        np.save(f"{PATH_VIOLIN_DATA}{self.standard_name}_{unit}.npy", violin_data)


class RandomQGrid(Grid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, gen_alg="randomQ", **kwargs)

    def generate_grid(self):
        result = np.zeros((self.N, 4))
        random_num = self.rn_gen.random((self.N, 3))
        result[:, 0] = np.sqrt(1 - random_num[:, 0]) * np.sin(2 * pi * random_num[:, 1])
        result[:, 1] = np.sqrt(1 - random_num[:, 0]) * np.cos(2 * pi * random_num[:, 1])
        result[:, 2] = np.sqrt(random_num[:, 0]) * np.sin(2 * pi * random_num[:, 2])
        result[:, 3] = np.sqrt(random_num[:, 0]) * np.cos(2 * pi * random_num[:, 2])
        self.grid = quaternion2grid(result)
        # No super call because ordering not needed for random points and the number of points is exact!


class RandomEGrid(Grid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, gen_alg="randomE", **kwargs)

    def generate_grid(self):
        euler_angles = 2 * pi * self.rn_gen.random((self.N, 3))
        self.grid = euler2grid(euler_angles)
        # No super call because ordering not needed for random points and the number of points is exact!


class SystemEGrid(Grid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, gen_alg="systemE", **kwargs)

    def generate_grid(self):
        self.grid = []
        num_points = 1
        while len(self.grid) < self.N:
            phis = np.linspace(0, 2 * pi, num_points)
            thetas = np.linspace(0, 2 * pi, num_points)
            psis = np.linspace(0, 2 * pi, num_points)
            euler_meshgrid = np.array(np.meshgrid(*(phis, thetas, psis)), dtype=float)
            euler_meshgrid = euler_meshgrid.reshape((3, -1)).T
            # convert to a grid
            self.grid = euler2grid(euler_meshgrid)
            self.grid = np.unique(np.round(self.grid, UNIQUE_TOL), axis=0)
            num_points += 1
        super().generate_grid()


class Cube4DGrid(Grid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, gen_alg="cube4D", **kwargs)
        np.random.seed(1)

    def generate_grid(self):
        self.grid = []
        num_divisions = 1
        state_before = np.random.get_state()
        while len(self.grid) < self.N:
            grid_qua = cube_grid_on_sphere(num_divisions, 4)
            grid_qua = select_half_sphere(grid_qua)
            # convert to grid
            self.grid = quaternion2grid(grid_qua)
            self.grid = np.unique(np.round(self.grid, UNIQUE_TOL), axis=0)
            num_divisions += 1
        np.random.set_state(state_before)
        super().generate_grid()


class Polyhedron3DGrid(Grid):

    def __init__(self, N: int, polyhedron, **kwargs):
        self.polyhedron = polyhedron()
        super().__init__(N, **kwargs)

    def generate_grid(self):
        while self.polyhedron.G.number_of_nodes() < self.N:
            self.polyhedron.divide_edges()
        self.grid = np.array([y["projection"] for x, y in self.polyhedron.G.nodes(data=True)]).squeeze()
        np.random.shuffle(self.grid)
        super().generate_grid()


class Cube3DGrid(Polyhedron3DGrid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, polyhedron=CubePolytope, gen_alg="cube3D", **kwargs)


class IcoGrid(Polyhedron3DGrid):

    def __init__(self, N: int, **kwargs):
        super().__init__(N, polyhedron=IcosahedronPolytope, gen_alg="ico", **kwargs)


def build_grid(grid_type: str, N: int, **kwargs) -> Grid:
    name2grid = {"randomQ": RandomQGrid,
                 "randomE": RandomEGrid,
                 "cube4D": Cube4DGrid,
                 "systemE": SystemEGrid,
                 "cube3D": Cube3DGrid,
                 "ico": IcoGrid}
    if grid_type not in name2grid.keys():
        raise ValueError(f"{grid_type} is not a valid grid type. Try 'ico', 'cube3D' ...")
    grid_obj = name2grid[grid_type]
    return grid_obj(N, **kwargs)


def build_all_grids(N: int, **kwargs) -> list:
    all_grids = []
    for grid_type in SIX_METHOD_NAMES:
        all_grids.append(build_grid(grid_type, N, **kwargs))
    return all_grids