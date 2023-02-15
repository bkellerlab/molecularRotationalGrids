import os

import numpy as np
import pandas as pd

from molgri.molecules.parsers import PtParser
from molgri.plotting.molecule_plots import TrajectoryPlot
from molgri.plotting.space_plots import SphereGridPlot, PolytopePlot, PanelSphereGridPlots
from molgri.space.cells import save_voranoi_data_for_alg
from molgri.space.fullgrid import FullGrid
from molgri.constants import SMALL_NS, GRID_ALGORITHMS, PATH_EXAMPLES
# from molgri.plotting.grid_plots import PositionGridPlot, create_hammer_multiplot, create_trajectory_energy_multiplot, groupby_min_body_energy
# from molgri.plotting.analysis_plots import VoranoiConvergencePlot
from molgri.space.polytopes import Cube3DPolytope, Cube4DPolytope, IcosahedronPolytope
from molgri.space.rotobj import SphereGridFactory


# def test_groupby_min_body_energy():
#     # translations already filtered out, n_b = 2, n_o = 3
#     test_arr = np.array([[1, 3, -2], [7, 8, 5], [1, -1, -8], [2, 1, 3], [-1.7, -0.3, -0.3], [8, 8, 5]])
#     start_df = pd.DataFrame(test_arr, columns=["x", "y", "E"])
#     end_df = groupby_min_body_energy(start_df, "E", 3)
#     expected_array = np.array([[1, -1, -8], [-1.7, -0.3, -0.3]])
#     expected_df = pd.DataFrame(expected_array, columns=["x", "y", "E"])
#     assert np.allclose(end_df, expected_df)


# def test_everything_runs():
#     # create the needed file
#     FullGrid(o_grid_name="cube3D_9", b_grid_name="zero", t_grid_name='range(1, 5, 2)')
#     PositionGridPlot("position_grid_o_cube3D_9_b_zero_1_t_3203903466", cell_lines=True).create_and_save(
#         animate_rot=True, animate_seq=True)
#     # create the needed file
#     save_voranoi_data_for_alg(alg_name="randomE", N_set=SMALL_NS, radius=1)
#     VoranoiConvergencePlot("randomE_1_8_300").create_and_save()
#     create_trajectory_energy_multiplot("H2O_H2O_o_ico_500_b_ico_5_t_3830884671", animate_rot=False)
#     create_hammer_multiplot("H2O_H2O_o_ico_500_b_ico_5_t_3830884671")


def test_polytope_plots():
    for pol in (Cube3DPolytope(), Cube4DPolytope(), IcosahedronPolytope()):
        pol.divide_edges()
        pp = PolytopePlot(pol)
        pp.create_all_plots()


def test_space_plots(N=12):
    for alg in GRID_ALGORITHMS[:-1]:
        for dim in (3, 4):
            sgf = SphereGridFactory.create(alg_name=alg, N=N, dimensions=dim,
                                           print_messages=False, time_generation=False,
                                           use_saved=False)
            SphereGridPlot(sgf).create_all_plots(and_animations=False)


def test_space_multi_plots(N=25):
    for dim in (3, 4):
        psgp = PanelSphereGridPlots(N, grid_dim=dim, default_context="talk")
        psgp.make_all_grid_plots(animate_rot=True)
        psgp.make_all_uniformity_plots()
        psgp.make_all_convergence_plots()


def test_trajectory_plots():
    # test with a pseudo-trajectory
    topology_path = os.path.join(PATH_EXAMPLES, "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.gro")
    trajectory_path = os.path.join(PATH_EXAMPLES, "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xtc")
    water_path = os.path.join(PATH_EXAMPLES, "H2O.gro")
    energy_path = os.path.join(PATH_EXAMPLES, "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xvg")
    parser_pt = PtParser(m1_path=water_path, m2_path=water_path, path_topology=topology_path,
                         path_trajectory=trajectory_path, path_energy=energy_path)
    TrajectoryPlot(parser_pt).create_all_plots(and_animations=False)
    # TODO: test with a simulated trajectory


if __name__ == "__main__":
    test_polytope_plots()
    test_trajectory_plots()
    test_space_multi_plots(N=200)
    test_space_plots(N=100)
