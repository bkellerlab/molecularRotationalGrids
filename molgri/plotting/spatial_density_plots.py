import MDAnalysis as mda
from MDAnalysis.analysis import density

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.express as px
from scipy import sparse
from plotly.subplots import make_subplots

WIDTH = 600
HEIGHT = 600
NUM_EIGENV = 6


class SpatialDensity:

    def __init__(self, path_structure, path_trajectory, path_density):
        self.trajectory_universe = mda.Universe(path_structure, path_trajectory)
        self.density_path = path_density
        pio.templates.default = "simple_white"
        self.fig = go.Figure()

    def show_isosurface_particle(self, atom_selection):
        ow = self.trajectory_universe.select_atoms(atom_selection)
        print(ow)
        D = density.DensityAnalysis(ow, delta=0.05)
        D.run()
        #D.density.convert_density('TIP4P')
        # ensure that the density is A^{-3}
        D.results.density.convert_density("A^{-3}")

        dV = np.prod(D.results.density.delta)
        atom_count_histogram = D.results.density.grid * dV
        D.results.density.export(self.density_path, type="double")

if __name__ == "__main__":
    my_str = f"experiments/msm_tau_ex/structure.gro"
    my_traj = f"experiments/msm_tau_ex/trajectory.xtc"
    my_density = f"experiments/msm_tau_ex/water.dx"

    sd = SpatialDensity(my_str, my_traj, my_density)
    sd.show_isosurface_particle("index 5")