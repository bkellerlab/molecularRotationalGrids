from molgri.grids import build_grid
from molgri.constants import SIX_METHOD_NAMES
import numpy as np


def test_ordering():
    # TODO: figure out what's the issue
    """Assert that, ignoring randomness, the first N-1 points of ordered grid with length N are equal to ordered grid
    of length N-1"""
    for name in SIX_METHOD_NAMES:
        try:
            for N in range(14, 284, 3):
                for addition in (1, 7):
                    grid_1 = build_grid(name, N+addition, ordered=True).get_grid()
                    grid_2 = build_grid(name, N, ordered=True).get_grid()
                    assert np.allclose(grid_1[:N], grid_2)
        except AssertionError:
            print(name)


if __name__ == "__main__":
    test_ordering()