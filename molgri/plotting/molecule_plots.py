import numpy as np
import matplotlib.cm as cm
import pandas as pd
from matplotlib.colors import Normalize
from numpy._typing import NDArray

from molgri.molecules.parsers import FileParser, PtParser
from molgri.plotting.abstract import RepresentationCollection


class TrajectoryPlot(RepresentationCollection):

    def __init__(self, parsed_file: FileParser):
        self.parsed_file = parsed_file
        data_name = self.parsed_file.get_topology_file_name()
        super().__init__(data_name)

    def _default_atom_selection(self, atom_selection: str):
        if atom_selection is None and isinstance(self.parsed_file, PtParser):
            atom_selection = self.parsed_file.get_atom_selection_r()
        return atom_selection

    def _make_scatter_plot(self, projection, data, **kwargs):
        self.ax.scatter(*data, **kwargs)
        if projection == "3d":
            self._equalize_axes()
        elif projection == "hammer":
            self.ax.set_xticks([])

    def make_COM_plot(self, ax=None, fig=None, save=True, atom_selection=None, projection="3d", animate_rot=False):
        self._create_fig_ax(ax=ax, fig=fig, projection=projection)
        # if a pseudotrajectory, default setting is to plot COM or r_molecule
        atom_selection = self._default_atom_selection(atom_selection)

        # get data
        coms = self.parsed_file.get_all_COM(atom_selection)
        # filter out unique ones
        coms, _ = get_unique_com(coms, None)

        # plot data
        self._make_scatter_plot(projection, coms.T, color="black")

        if save:
            self._save_plot_type(f"com_{projection}")

        if animate_rot and projection == "3d":
            self._animate_figure_view(self.fig, self.ax, f"com_rotated")

    def make_energy_COM_plot(self, ax=None, fig=None, save=True, atom_selection=None, projection="3d",
                             animate_rot=False):
        self._create_fig_ax(ax=ax, fig=fig, projection=projection)
        atom_selection = self._default_atom_selection(atom_selection)

        # get energies
        energies = self.parsed_file.get_all_energies("Potential")
        if energies is None:
            print(f"This energy_COM plot could not be plotted since energies of {self.data_name} were not found.")
            return
        coms = self.parsed_file.get_all_COM(atom_selection)
        # filter for lowest energy per point
        coms, energies = get_unique_com(coms, energies)

        # convert energies to colors
        cmap = cm.coolwarm
        norm = Normalize(vmin=np.min(energies), vmax=np.max(energies))

        # plot data
        self._make_scatter_plot(projection, coms.T, cmap=cmap, c=energies, norm=norm)

        if save:
            self._save_plot_type(f"energies_{projection}")

        if animate_rot and projection == "3d":
            self._animate_figure_view(self.fig, self.ax, f"energies_rotated")

    def create_all_plots(self, and_animations=False):
        self.make_COM_plot(projection="3d", animate_rot=and_animations)
        self.make_COM_plot(projection="hammer")
        self.make_energy_COM_plot(projection="3d", animate_rot=and_animations)
        self.make_energy_COM_plot(projection="hammer")


def get_unique_com(coms: NDArray, energies: NDArray = None):
    """
    Get only the subset of COMs that have unique positions. Among those, select the ones with lowest energy (if
    energy info is provided)
    """
    round_to = 3  # number of decimal places
    if energies is None:
        _, indices = np.unique(coms.round(round_to), axis=0, return_index=True)
        unique_coms = np.take(coms, indices, axis=0)
        return unique_coms, energies

    # if there are energies, among same COMs, select the one with lowest energy
    coms_tuples = [tuple(row.round(round_to)) for row in coms]
    df = pd.DataFrame()
    df["coms_tuples"] = coms_tuples
    df["energy"] = energies
    new_df = df.loc[df.groupby(df["coms_tuples"])["energy"].idxmin()]
    unique_coms = np.take(coms, new_df.index, axis=0)
    unique_energies = np.take(energies, new_df.index, axis=0)
    return unique_coms, unique_energies


if __name__ == "__main__":
    import os

    topology_path = os.path.join("output", "pt_files", "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.gro")
    trajectory_path = os.path.join("output", "pt_files", "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xtc")
    water_path = os.path.join("input", "base_gro_files", "H2O.gro")
    parser_pt = PtParser(m1_path=water_path, m2_path=water_path, path_topology=topology_path, path_trajectory=trajectory_path)
    TrajectoryPlot(parser_pt).create_all_plots(and_animations=True)