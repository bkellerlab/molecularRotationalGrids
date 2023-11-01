
from molgri.molecules._load_examples import load_example_pt
from molgri.plotting.molecule_plots import ConvergenceMultiCollectionPlot, TrajectoryPlot
from molgri.plotting.spheregrid_plots import EightCellsPlot, SphereGridPlot, PolytopePlot, PanelSphereGridPlots, \
    ConvergenceSphereGridPlot
from molgri.plotting.fullgrid_plots import PositionGridPlot, ConvergenceFullGridPlot
from molgri.constants import (DEFAULT_ALGORITHM_B, DEFAULT_ALGORITHM_O, MINI_NS, GRID_ALGORITHMS_3D,
                              GRID_ALGORITHMS_4D)
from molgri.space.fullgrid import FullGrid, ConvergenceFullGridO
from molgri.space.polytopes import Cube3DPolytope, Cube4DPolytope, IcosahedronPolytope
from molgri.space.rotobj import ConvergenceSphereGridFactory, SphereGrid3DFactory, \
    SphereGrid4DFactory



def test_polytope_plots(and_animations=False):
    for pol in (Cube3DPolytope(), IcosahedronPolytope(), Cube4DPolytope()):
        pol.divide_edges()
        pp = PolytopePlot(pol, default_complexity_level="empty")
        pp.create_all_plots(and_animations=and_animations)

    for only_half_of_cube in (False, True):
        cube4D = Cube4DPolytope()
        cube4D.divide_edges()
        ecp = EightCellsPlot(cube4D, only_half_of_cube=only_half_of_cube)
        ecp.create_all_plots()


def test_spheregrid_plots(N=32, and_animations=False, Ns=MINI_NS):
    # single plots 3D
    for alg in GRID_ALGORITHMS_3D:
        # grid plotting
        sgf = SphereGrid3DFactory.create(alg_name=alg, N=N, time_generation=False,
                                         use_saved=False)
        SphereGridPlot(sgf).create_all_plots(and_animations=and_animations)
    # single plots 4D
    # fulldiv will not work for a non-allowed num of points
    # for alg in GRID_ALGORITHMS_4D[:-1]:
    #     # grid plotting
    #     sgf = SphereGrid4DFactory.create(alg_name=alg, N=N, time_generation=False,
    #                                      use_saved=False)
    #     SphereGridPlot(sgf).create_all_plots(and_animations=and_animations)

    # panel plots
    psgp = PanelSphereGridPlots(N, 3)
    psgp.create_all_plots()
    psgp = PanelSphereGridPlots(N, 4)
    psgp.create_all_plots()

    # convergence plots
    csg = ConvergenceSphereGridFactory("ico", 3, N_set=Ns)
    psgp = ConvergenceSphereGridPlot(csg)
    psgp.create_all_plots()
    csg2 = ConvergenceSphereGridFactory("cube4D", 4, N_set=Ns)
    psgp2 = ConvergenceSphereGridPlot(csg2)
    psgp2.create_all_plots()


def test_fullgrid_plots(N=12, and_animations=False, N_set=MINI_NS):
    # just set up one example
    fg = FullGrid(f"{DEFAULT_ALGORITHM_B}_{N}", f"{DEFAULT_ALGORITHM_O}_{N}", "[0.1, 0.2, 0.3]", use_saved=False)
    fgp = PositionGridPlot(fg)
    fgp.create_all_plots(and_animations=and_animations)

    cgrid = ConvergenceFullGridO("zero", "[0.1, 0.2]", "ico", MINI_NS)
    ConvergenceFullGridPlot(cgrid).create_all_plots()


# def test_transition_plots(and_animations=False):
#     # TODO: need to do this with small, local examples
#
#     # rate matrix example
#     # doesn't work currently but that's more of data issue
#
#     # molgri_pt = load_molgri_data()
#     # full_grid_m = FullGrid(t_grid_name="linspace(0.8, 1.5, 10)", o_grid_name="ico_50", b_grid_name="zero")
#     # combined_molgri = SimulationHistogram(molgri_pt, full_grid_m)
#     # sqra = SQRA(combined_molgri, energy_type="Potential Energy", use_saved=False)
#     # tp_sqra = TransitionPlot(sqra, default_context="talk")
#     # tp_sqra.create_all_plots(and_animations=True)
#
#     # MSM example
#     # parsed_sim = load_simulation_data()
#     # # define some full grid to assign to
#     # full_grid = FullGrid(t_grid_name="linspace(3, 13, 4)", o_grid_name="ico_20", b_grid_name="zero")
#     # combined_sim = SimulationHistogram(parsed_sim, full_grid)
#     # msm = MSM(combined_sim, use_saved=True)
#     # tp_msm = TransitionPlot(msm, default_context="talk")
#     # tp_msm.create_all_plots(and_animations=and_animations)


def test_trajectory_plots(and_animations=False):
    # test with a pseudo-trajectory
    pt = load_example_pt()
    TrajectoryPlot(pt).create_all_plots(and_animations=and_animations)
    # # default Ns
    ConvergenceMultiCollectionPlot(pt, MINI_NS).create_all_plots()


if __name__ == "__main__":
    and_animations = False
    test_polytope_plots(and_animations=and_animations)
    test_spheregrid_plots(and_animations=and_animations)
    test_fullgrid_plots(and_animations=and_animations)
    test_trajectory_plots(and_animations=and_animations)