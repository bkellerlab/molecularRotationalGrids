import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from matplotlib.pyplot import Figure, Axes
import matplotlib.colors as colors

from molgri.plotting.abstract import RepresentationCollection


class ArrayPlot(RepresentationCollection):

    """
    A tool for plotting arrays, eg by highlighting high and low values
    """

    def __init__(self, my_array: NDArray, *args, data_name="array", **kwargs):
        self.array = my_array
        super().__init__(data_name, *args, **kwargs)

    def make_heatmap_plot(self, ax: Axes = None, fig: Figure = None, save: bool = True):
        """
        This method draws the array and colors the fields according to their values (red = very large,
        blue = very small). Zero values are always white, negative ones always blue, positive ones always red.
        """
        self._create_fig_ax(fig=fig, ax=ax)
        if np.all(self.array < 0):
            cmap = "Blues"
            norm = None
        elif np.all(self.array > 0):
            cmap = "Reds"
            norm = None
        else:
            cmap = "bwr"
            norm = colors.TwoSlopeNorm(vcenter=0)
        sns.heatmap(self.array, cmap=cmap, ax=self.ax, xticklabels=False, yticklabels=False, norm=norm)

        self._equalize_axes()
        if save:
            self._save_plot_type("heatmap")


if __name__ == "__main__":
    import pandas as pd
    from molgri.molecules.transitions import SimulationHistogram, SQRA, MSM
    from molgri.molecules.parsers import ParsedEnergy, FileParser
    from molgri.space.fullgrid import FullGrid

    path_energy = "/home/mdglasius/Modelling/trypsin_test/output/measurements/temp_minimized.csv"
    df = pd.read_csv(path_energy)
    energies = df['potential'].to_numpy()[:, np.newaxis]
    pe = ParsedEnergy(energies=energies, labels=["Potential"], unit="(kJ/mole)")
    pt_parser = FileParser(
        path_topology="/home/mdglasius/Modelling/trypsin_test/output/pt_files/final_trypsin_NH4_o_ico_512_b_zero_1_t_2153868773.gro",
        path_trajectory="/home/mdglasius/Modelling/trypsin_test/output/pt_files/minimizedPT.pdb")
    parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="not protein")
    parsed_trajectory.energies = pe
    print(len(parsed_trajectory.get_all_COM()))

    # the exact grid used
    fg = FullGrid(t_grid_name="linspace(0.8, 2.5, 4)", o_grid_name="ico_10", b_grid_name="zero")

    # plotting
    sh = SimulationHistogram(parsed_trajectory, fg)
    my_sqra = SQRA(sh, use_saved=False)

    rate_matrix = my_sqra.get_transitions_matrix()

    # visualise rate matrix
    ArrayPlot(rate_matrix[0], data_name="rate_matrix").make_heatmap_plot()

    # TRANSITION MATRIX
    pt_parser = FileParser(
        path_topology="/home/mdglasius/Modelling/trypsin_normal/inputs/trypsin_probe.pdb",
        path_trajectory="/home/mdglasius/Modelling/trypsin_normal/nobackup/outputs/aligned_traj.dcd")
    parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="segid B")
    print(len(parsed_trajectory.get_all_COM()))

    # preparing the grid
    fg = FullGrid(t_grid_name="linspace(3, 13, 4)", o_grid_name="ico_40", b_grid_name="zero")

    sh = SimulationHistogram(parsed_trajectory, fg)
    my_msm = MSM(sh, use_saved=False)
    transition_matrix = my_msm.get_transitions_matrix()

    # visualise transition matrix
    ArrayPlot(transition_matrix[0], data_name="transition_matrix").make_heatmap_plot()
