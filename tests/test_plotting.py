import os
import numpy as np

from molgri.molecules.parsers import PtParser
from molgri.plotting.molecule_plots import TrajectoryPlot, ConvergenceMultiCollection
from molgri.plotting.spheregrid_plots import EightCellsPlot, SphereGridPlot, PolytopePlot, PanelSphereGridPlots, \
    ConvergenceSphereGridPlot, PanelConvergenceSphereGridPlots
from molgri.plotting.fullgrid_plots import FullGridPlot, ConvergenceFullGridPlot, PanelConvergenceFullGridPlots
from molgri.plotting.other_plots import ArrayPlot

from molgri.constants import ALL_GRID_ALGORITHMS, PATH_EXAMPLES, MINI_NS, GRID_ALGORITHMS_3D, GRID_ALGORITHMS_4D
from molgri.space.fullgrid import FullGrid, ConvergenceFullGridO
from molgri.space.polytopes import Cube3DPolytope, Cube4DPolytope, IcosahedronPolytope
from molgri.space.rotobj import SphereGridFactory, ConvergenceSphereGridFactory, SphereGrid3DFactory, \
    SphereGrid4DFactory


def get_example_pt():
    topology_path = os.path.join(PATH_EXAMPLES, "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.gro")
    trajectory_path = os.path.join(PATH_EXAMPLES, "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xtc")
    water_path = os.path.join(PATH_EXAMPLES, "H2O.gro")
    energy_path = os.path.join(PATH_EXAMPLES, "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xvg")
    parser_pt = PtParser(m1_path=water_path, m2_path=water_path, path_topology=topology_path,
                         path_trajectory=trajectory_path, path_energy=energy_path).get_parsed_trajectory()
    return parser_pt


def test_polytope_plots(animate_rot=False):
    for pol in (Cube3DPolytope(), IcosahedronPolytope(), Cube4DPolytope()):
        pol.divide_edges()
        pp = PolytopePlot(pol, default_complexity_level="empty")
        pp.create_all_plots(animate_rot=animate_rot)

    for only_half_of_cube in (False, True):
        cube4D = Cube4DPolytope()
        cube4D.divide_edges()
        ecp = EightCellsPlot(cube4D, only_half_of_cube=only_half_of_cube)
        #ecp.create_all_plots()


def test_spheregrid_plots(N=32, animate_rot=False, Ns=MINI_NS):
    # single plots 3D
    for alg in GRID_ALGORITHMS_3D:
        # grid plotting
        sgf = SphereGrid3DFactory.create(alg_name=alg, N=N, time_generation=False,
                                         use_saved=False)
        SphereGridPlot(sgf).create_all_plots(animate_rot=animate_rot)
    # single plots 4D
    for alg in GRID_ALGORITHMS_4D:
        # grid plotting
        sgf = SphereGrid4DFactory.create(alg_name=alg, N=N, time_generation=False,
                                         use_saved=False)
        SphereGridPlot(sgf).create_all_plots(animate_rot=animate_rot)

    # # convergence plotting
    # csg = ConvergenceSphereGridFactory(alg, dim, N_set=Ns)
    # csgp = ConvergenceSphereGridPlot(csg)
    # csgp.make_voronoi_area_conv_plot()
    # csgp.make_spheregrid_time_plot()
    # # panel plots
    # for dim in (3, 4):
    #     psgp = PanelSphereGridPlots(N, grid_dim=dim, default_context="talk")
    #     psgp.create_all_plots(and_animations=and_animations)
    #     pcsgp = PanelConvergenceSphereGridPlots(dim, N_set=Ns)
    #     pcsgp.make_all_voronoi_area_plots()
    #     pcsgp.make_all_spheregrid_time_plots()


def test_animation():
    # only test one to not overwhelm the system
    sgf = SphereGridFactory.create(alg_name="randomE", N=15, dimensions=3,
                                   print_messages=False, time_generation=False,
                                   use_saved=False)
    sgp = SphereGridPlot(sgf)
    sgp.make_rot_animation()
    sgp.make_trans_animation()
    sgp.make_ordering_animation()


def test_fullgrid_plots(N=12, and_animations=False, N_set=MINI_NS):
    t_grid_name = "[1, 3]"
    for alg in GRID_ALGORITHMS[:-1]:
        b_grid_name = f"{alg}_{N}"
        o_grid_name = f"{alg}_{2*N}"
        fg = FullGrid(b_grid_name=b_grid_name, o_grid_name=o_grid_name, t_grid_name=t_grid_name, use_saved=False)
        FullGridPlot(fg).create_all_plots(and_animations=and_animations)
        cfgo = ConvergenceFullGridO(b_grid_name=b_grid_name, t_grid_name=t_grid_name, o_alg_name=alg, N_set=N_set)
        ConvergenceFullGridPlot(cfgo).make_voronoi_volume_conv_plot()
    # panel plot
    PanelConvergenceFullGridPlots(t_grid_name=t_grid_name, N_set=N_set, use_saved=False).make_all_voronoi_volume_plots()


def test_trajectory_plots():
    # test with a pseudo-trajectory
    pt = get_example_pt()
    TrajectoryPlot(pt).create_all_plots(and_animations=False)
    # default Ns
    ConvergenceMultiCollection(pt).create_all_plots(and_animations=False)
    # user-defined Ns
    ConvergenceMultiCollection(pt, N_set=(10, 20, 50)).create_all_plots(and_animations=False)


def test_other_plots():
    my_array = np.array([[-3, 2, 0, 7],
                         [2, 1, -17, 0],
                         [2, 1, 0, -13],
                         [2, 2, 2, 2]])
    ArrayPlot(my_array).make_heatmap_plot()

if __name__ == "__main__":
    #test_polytope_plots(animate_rot=False)
    test_spheregrid_plots(animate_rot=False)