"""
Plots connected with (pseudo)trajectories and their energies.

Enables plotting COM of ParsedTrajectory and coloring them according to energies, convergence of energies with the
length of the (pseudo)trajectory.
"""

import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns

from molgri.constants import ENERGY2SHORT
from molgri.molecules.parsers import ParsedTrajectory
from molgri.molecules.transitions import SimulationHistogram
from molgri.plotting.abstract import RepresentationCollection, MultiRepresentationCollection
from molgri.plotting.fullgrid_plots import FullGridPlot
from molgri.wrappers import plot_method, plot3D_method


class TrajectoryPlot(RepresentationCollection):

    def __init__(self, parsed_trajectory: ParsedTrajectory, N_used: int = None):
        self.parsed_trajectory = parsed_trajectory
        if N_used is None:
            N_used = None
        self.N_used = N_used
        data_name = self.parsed_trajectory.get_name()
        super().__init__(data_name)

    def _default_atom_selection(self, atom_selection: str):
        if atom_selection is None and self.parsed_trajectory.is_pt:
            atom_selection = self.parsed_trajectory.get_atom_selection_r()
        return atom_selection

    def _make_scatter_plot(self, projection, data, **kwargs):
        sc = self.ax.scatter(*data, **kwargs)
        if projection == "3d":
            self._equalize_axes()
        elif projection == "hammer":
            self.ax.set_xticks([])
        return sc

    def get_possible_title(self):
        return f"N={self.N_used}"

    @plot3D_method
    def plot_atoms(self, atom_selection=None):
        # if a pseudotrajectory, default setting is to plot COM or r_molecule
        atom_selection = self._default_atom_selection(atom_selection)

        for mol in self.parsed_trajectory.molecule_generator(atom_selection):
            for atom in mol.get_atoms():
                if "O" in atom.type:
                    color="red"
                elif "H" in atom.type:
                    color = "gray"
                else:
                    color="blue"
                self.ax.scatter(*atom.position, color=color)
        self._equalize_axes()

    @plot3D_method
    def plot_COM(self, atom_selection=None, fg = None, projection="3d", **kwargs):
        # if a pseudotrajectory, default setting is to plot COM or r_molecule
        atom_selection = self._default_atom_selection(atom_selection)


        # filter out unique COMs
        coms, _ = self.parsed_trajectory.get_unique_com_till_N(N=self.N_used, atom_selection=atom_selection)
        if fg and projection == "3d":
            fgp = FullGridPlot(fg)
            fgp.plot_position_voronoi(ax=self.ax, fig=self.fig, animate_rot=False, plot_vertex_points=False,
                                      save=False, numbered=False)
            _, c = self.parsed_trajectory.assign_coms_2_grid_points(full_grid=fg, atom_selection=atom_selection,
                                                                 coms=coms)
            cmap="Spectral"
        else:
            c="black"
            cmap=None
        # plot data
        self._make_scatter_plot(projection, coms.T, c=c, cmap=cmap)

    @plot3D_method
    def plot_energy_COM(self, atom_selection=None, projection="3d", energy_type="Potential", vmin=None, vmax=None,
                        lowest_k = None, highest_j = None, **kwargs):
        atom_selection = self._default_atom_selection(atom_selection)

        coms, energies = self.parsed_trajectory.get_unique_com_till_N(N=self.N_used, energy_type=energy_type,
                                                                      atom_selection=atom_selection)
        coms, energies = self.parsed_trajectory.get_only_lowest_highest(coms, energies, lowest_k=lowest_k,
                                                                        highest_j=highest_j)
        # convert energies to colors
        cmap = cm.coolwarm
        if vmin is None:
            vmin = np.min(energies)
        if vmax is None:
            vmax = np.max(energies)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # plot data
        sc = self._make_scatter_plot(projection, coms.T, cmap=cmap, c=energies, norm=norm)

        try:
            save_name = f"energies_{ENERGY2SHORT[energy_type]}"
        except KeyError:
            save_name = "energies"
        self.fig.colorbar(sc, ax=self.ax)

    @plot_method
    def plot_energy_violin(self, energy_type: str = "Potential", **kwargs):
        coms, energies = self.parsed_trajectory.get_unique_com_till_N(N=self.N_used, energy_type=energy_type)
        sns.violinplot(energies, ax=self.ax, scale="count", inner="stick", cut=0)
        self.ax.set_xticklabels([])
        self.ax.set_ylabel(f"{energy_type} [kJ/mol]")
