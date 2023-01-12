import numpy as np
from numpy.typing import NDArray

from molgri.parsers import TranslationParser, GridNameParser
from molgri.paths import PATH_OUTPUT_FULL_GRIDS
from molgri.rotobj import build_rotations_from_name
from molgri.utils import norm_per_axis


class FullGrid:

    def __init__(self, b_grid_name: str, o_grid_name: str, t_grid_name: str):
        """
        A combination object that enables work with a set of grids. A parser that

        Args:
            b_grid_name: body rotation grid
            o_grid_name: origin rotation grid
            t_grid_name: translation grid
        """
        b_grid_name = GridNameParser(b_grid_name, "b").get_standard_grid_name()
        self.b_rotations = build_rotations_from_name(b_grid_name, "b")
        o_grid_name = GridNameParser(o_grid_name, "o").get_standard_grid_name()
        self.o_rotations = build_rotations_from_name(o_grid_name)
        self.o_name = self.o_rotations.standard_name
        self.o_positions = self.o_rotations.get_grid_z_as_array()
        self.t_grid = TranslationParser(t_grid_name)
        self.save_full_grid()

    def get_full_grid_name(self):
        return f"o_{self.o_name}_b_{self.b_rotations.standard_name}_t_{self.t_grid.grid_hash}"

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


if __name__ == "__main__":
    fg = FullGrid(o_grid_name="randomQ_7", t_grid_name="[1, 2, 3]", b_grid_name="randomQ_14")
    import matplotlib.pyplot as plt
    import seaborn as sns

    pos_grid = fg.get_position_grid()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # print(pos_grid[0].shape)
    # ax.scatter(*pos_grid[0].T, c="r")
    # ax.scatter(*pos_grid[1].T, c="b")
    # ax.scatter(*pos_grid[2].T, c="g")
    # plt.show()
    pos_grid = np.swapaxes(pos_grid, 0, 1)
    pos_grid = pos_grid.reshape((-1, 3))
    rgb_values = sns.color_palette("flare", len(pos_grid))
    for i, el in enumerate(pos_grid):
        ax.scatter(*el, c=rgb_values[i], label=i)
        ax.text(*el, i)
    plt.show()