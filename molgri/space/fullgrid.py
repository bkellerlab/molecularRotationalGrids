import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from molgri.space.rotobj import SphereGridFactory
from molgri.molecules.parsers import TranslationParser, GridNameParser
from molgri.paths import PATH_OUTPUT_FULL_GRIDS
from molgri.space.utils import norm_per_axis


class FullGrid:

    def __init__(self, b_grid_name: str, o_grid_name: str, t_grid_name: str, use_saved: bool = True):
        """
        A combination object that enables work with a set of grids. A parser that

        Args:
            b_grid_name: body rotation grid (should be made into a 4D sphere grid)
            o_grid_name: origin rotation grid (should be made into a 3D sphere grid)
            t_grid_name: translation grid
        """
        b_grid_name = GridNameParser(b_grid_name, "b")
        self.b_rotations = SphereGridFactory.create(alg_name=b_grid_name.get_alg(), N=b_grid_name.get_N(),
                                                    dimensions=4, use_saved=use_saved)
        o_grid_name = GridNameParser(o_grid_name, "o")
        self.o_rotations = SphereGridFactory.create(alg_name=o_grid_name.get_alg(), N=o_grid_name.get_N(),
                                                    dimensions=3, use_saved=use_saved)
        self.o_positions = self.o_rotations.get_grid_as_array()
        self.t_grid = TranslationParser(t_grid_name)
        self.save_full_grid()

    def get_full_grid_name(self):
        o_name = self.o_rotations.get_standard_name(with_dim=False)
        b_name = self.b_rotations.get_standard_name(with_dim=False)
        return f"o_{o_name}_b_{b_name}_t_{self.t_grid.grid_hash}"

    def get_body_rotations(self) -> Rotation:
        return Rotation.from_quat(self.b_rotations.get_grid_as_array())

    def get_position_grid(self) -> NDArray:
        """
        Get a 'product' of o_grid and t_grid so you can visualise points in space at which COM of the second molecule
        will be positioned. Important: Those are points on spheres in real space.

        Returns:
            an array of shape (len_o_grid, len_t_grid, 3) in which the elements of result[0] have the first
            rotational position, each line at a new (increasing) distance, result[1] the next rotational position,
            again at all possible distances ...
        """
        dist_array = self.t_grid.get_trans_grid()
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

    def save_full_grid(self):
        np.save(f"{PATH_OUTPUT_FULL_GRIDS}position_grid_{self.get_full_grid_name()}", self.get_position_grid())
