from scipy.constants import pi
import numpy as np

from molgri.grids import FullGrid
from molgri.plotting import GridColoredWithAlphaPlot, GridPlot, AlphaViolinPlot, AlphaConvergencePlot, PolytopePlot, \
    PositionGridPlot, VoranoiConvergencePlot


def test_everything_runs():
    # examples of grid plots and animations
    GridColoredWithAlphaPlot("ico_85", vector=np.array([0, 0, 1]), alpha_set=[pi/6, pi/3, pi/2, 2*pi/3],
                             style_type=None).create_and_save()
    GridPlot("ico_22").create_and_save(title="Icosahedron, 22 points", x_label="x", y_label="y", z_label="z", animate_rot=True,
                              animate_seq=True, main_ticks_only=True)
    GridPlot("cube3D_500", style_type=["talk", "empty"]).create_and_save()
    GridPlot("cube3D_12", style_type=["talk", "half_dark"]).create_and_save()
    GridPlot("ico_22", style_type=["talk", "dark"]).create_and_save()
    # examples of statistics/convergence plots
    AlphaViolinPlot("ico_250").create_and_save(title="ico grid, 250")
    AlphaConvergencePlot("systemE", style_type=["talk"]).create_and_save(equalize=True, title="Convergence of systemE")
    AlphaConvergencePlot("ico_17", style_type=None).create_and_save(title="Convergence of ico", main_ticks_only=True)
    # examples of polyhedra
    PolytopePlot("ico", num_divisions=2, faces={0, 1, 2, 3, 4}).create_and_save(equalize=True, elev=190, azim=120,
                                                                       x_max_limit=0.55, x_min_limit=-0.6)
    PolytopePlot("ico", num_divisions=2, projection=True).create_and_save(equalize=True, elev=190, azim=120,
                                                                 x_max_limit=0.55, x_min_limit=-0.6)
    PolytopePlot("cube3D", num_divisions=3).create_and_save(equalize=True, elev=0, azim=0, x_max_limit=0.7, x_min_limit=-0.7)
    FullGrid(o_grid_name="cube3D_9", b_grid_name="zero", t_grid_name='range(1, 5, 2)')
    PositionGridPlot("position_grid_o_cube3D_9_b_zero_1_t_3203903466", cell_lines=True).create_and_save(
        animate_rot=True, animate_seq=True)
    VoranoiConvergencePlot("cube4D_1_10_100").create_and_save()


if __name__ == "__main__":
    test_everything_runs()