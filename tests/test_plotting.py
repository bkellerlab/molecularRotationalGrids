from scipy.constants import pi
import numpy as np

from molgri.plotting import GridColoredWithAlphaPlot, GridPlot, AlphaViolinPlot, AlphaConvergencePlot, PolytopePlot


def test_everything_runs():
    # examples of grid plots and animations
    GridColoredWithAlphaPlot("ico_85", vector=np.array([0, 0, 1]), alpha_set=[pi/6, pi/3, pi/2, 2*pi/3],
                             style_type=["talk", "empty"]).create()
    GridPlot("ico_22").create(title="Icosahedron, 22 points", x_label="x", y_label="y", z_label="z", animate_rot=True,
                              animate_seq=True, main_ticks_only=True)
    GridPlot("cube3D_500", style_type=["talk", "empty"]).create()
    # examples of statistics/convergence plots
    AlphaViolinPlot("ico_250", style_type=["talk"]).create(title="ico grid, 250")
    AlphaConvergencePlot("systemE", style_type=["talk"]).create(title="Convergence of systemE")
    # examples of polyhedra
    PolytopePlot("ico", num_divisions=2, faces={0, 1, 2, 3, 4}).create(equalize=True, elev=190, azim=120,
                                                                       pos_limit=0.55, neg_limit=-0.6)
    PolytopePlot("cube3D", num_divisions=3).create(equalize=True, elev=0, azim=0, pos_limit=0.7, neg_limit=-0.7)
