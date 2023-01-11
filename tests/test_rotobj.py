
from molgri.rotobj import build_rotations
from molgri.constants import GRID_ALGORITHMS

import numpy as np


def test_rotobj2grid2rotobj():
    for algo in GRID_ALGORITHMS:
        for N in (12, 23, 51):
            rotobj_start = build_rotations(N, algo, use_saved=False)
            matrices_start = rotobj_start.rotations.as_matrix()
            rotobj_new = build_rotations(N, algo, use_saved=True)
            matrices_new = rotobj_new.rotations.as_matrix()
            assert np.allclose(matrices_start, matrices_new)