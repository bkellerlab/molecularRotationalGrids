import os

import numpy as np
import pandas as pd

from molgri.constants import DEFAULT_ALGORITHM_O, DEFAULT_ALGORITHM_B, ZERO_ALGORITHM, DEFAULT_ALPHAS_3D, \
    DEFAULT_ALPHAS_4D, DEFAULT_NS
from molgri.space.rotobj import RandomQRotations, SystemERotations, RandomERotations, Cube4DRotations, ZeroRotations, \
    IcoRotations, Cube3DRotations, RotationsObject


class SpaceParser:
    """
    This object deals with parsing inputs connected to rotational grids - should raise errors if critical details
    are missing or select defaults if non-critical details are missing.

    Should be able to input keywords like 'ico' and '500' and with appropriate getters obtain:
     - 4D hypersphere grids
     - 3D sphere grids
     - corresponding uniformity analysis df (for both)
     - corresponding convergence analysis df (for both)
     - corresponding polyhedron, if exists

     The idea is that all plotting functions (and FullGrid) should only rely on those getters.
    """

    def __init__(self, N: int, alg_name: str, dimension: int, use_saved_data: bool):
        # use defaults; explicitly pass None if defaults are to be used
        assert dimension == 3 or dimension == 4, f"Can only generate space grids with 3 or 4 dimensions not {dimension}"
        if alg_name is None and N == 1:
            alg_name = ZERO_ALGORITHM
        elif alg_name is None and dimension == 3:
            alg_name = DEFAULT_ALGORITHM_O
        elif alg_name is None and dimension == 4:
            alg_name = DEFAULT_ALGORITHM_B
        if use_saved_data is None:
            use_saved_data = True
        self.N = N
        self.alg_name = alg_name
        self.dimension = dimension
        self.use_saved_data = use_saved_data

    def _create_rotobj(self):
        assert self.dimension == 4, "Cannot create hypergrid from initiated 3D object"
        assert self.N >= 1, f"N must be a positive integer, not {N}"
        rot_obj_factory = get_RotationsObject(self.alg_name)
        rot_obj = rot_obj_factory.__init__(N=self.N, gen_algorithm=self.alg_name, use_saved=self.use_saved_data)
        return rot_obj

    def get_hypergrid(self):
        rot_obj = self._create_rotobj()
        return rot_obj.as_quaternion()

    def _create_uniformity(self):
        """Common method for 3- and 4D."""

    def get_hypergrid_uniformity(self, alphas: tuple = DEFAULT_ALPHAS_4D):
        rot_obj = self._create_rotobj()
        # recalculate if: 1) self.use_saved_data = False OR 2) no saved data exists
        if not self.use_saved_data or not os.path.exists(rot_obj.statistics_path):
            rot_obj.save_statistics(alphas=alphas)
        ratios_df = pd.read_csv(rot_obj.statistics_path, dtype=float)
        # OR 3) provided alphas don't match those in the found file
        saved_alphas = set(ratios_df["alphas"])
        if saved_alphas != alphas:
            rot_obj.save_statistics(alphas=alphas)
            ratios_df = pd.read_csv(rot_obj.statistics_path, dtype=float)
        return ratios_df

    def get_hypergrid_convergence(self, alphas: tuple = DEFAULT_ALPHAS_4D, N_list: tuple = None):
        if self.N is None and N_list is None:
            N_list = np.array(DEFAULT_NS, dtype=int)
        elif self.N is not None:
            # create equally spaced convergence set
            assert self.N >= 3, f"N={self.N} not large enough to study convergence"
            N_list = np.logspace(np.log10(3), np.log10(self.N), dtype=int)
            N_list = np.unique(N_list)
        full_df = []
        for N in N_list:
            grid_factory = SpaceParser(N=N, alg_name=self.alg_name, dimension=self.dimension,
                                       use_saved_data=self.use_saved_data)
            df = grid_factory.get_hypergrid_uniformity(alphas=alphas)
            df["N"] = N
            full_df.append(df)
        full_df = pd.concat(full_df, axis=0, ignore_index=True)
        return full_df


def get_RotationsObject(alg_name) -> RotationsObject:
    name2rotation = {"randomQ": RandomQRotations,
                     "systemE": SystemERotations,
                     "randomE": RandomERotations,
                     "cube4D": Cube4DRotations,
                     "zero": ZeroRotations,
                     "ico": IcoRotations,
                     "cube3D": Cube3DRotations
                     }
    return name2rotation[alg_name]
