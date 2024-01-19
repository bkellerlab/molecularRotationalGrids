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


class ConvergenceMultiCollectionPlot(MultiRepresentationCollection):

    def __init__(self, parsed_trajectory: ParsedTrajectory, N_set: tuple = None):
        data_name = f"convergence_{parsed_trajectory.get_name()}"

        if N_set is None:
            N_max = parsed_trajectory.get_num_unique_com()
            N_set = np.linspace(N_max//10, N_max, num=5, dtype=int)
        self.N_set = N_set

        # generate TrajectoryPlots with limited number of points
        list_plots = [TrajectoryPlot(parsed_trajectory, N_used=N) for N in self.N_set]
        super().__init__(data_name, list_plots, n_columns=len(self.N_set), n_rows=1)

    def make_all_COM_3d_plots(self, animate_rot=False, save=True):

        self._make_plot_for_all("plot_COM", projection="3d", remove_midlabels=False,
                                creation_kwargs={"sharex": False, "sharey": False},
                                plotting_kwargs={"projection": "3d"})

        titles = [subplot.get_possible_title() for subplot in self.list_plots]
        self.add_titles(list_titles=titles, pad=-14)
        self.unify_axis_limits()

        if animate_rot:
            self.animate_figure_view(f"COM_3d")
        if save:
            self._save_multiplot(f"COM_3d")

    def make_all_COM_hammer_plots(self, save=True):
        self._make_plot_for_all("plot_COM", projection="hammer", remove_midlabels=False,
                                creation_kwargs={"sharex": False, "sharey": False},
                                plotting_kwargs={"projection": "hammer"})

        titles = [subplot.get_possible_title() for subplot in self.list_plots]
        self.add_titles(list_titles=titles)

        if save:
            self._save_multiplot(f"COM_hammer")

    def make_all_energy_plots(self, dim, animate_rot=False, save=True, energy_type: str = "Potential"):
        _, all_energies = self.list_plots[-1].parsed_trajectory.get_unique_com_till_N(N=np.max(self.N_set),
                                                                                   energy_type=energy_type)
        vmax = np.max(all_energies)
        vmin = np.min(all_energies)
        if dim == 3:
            projection = "3d"
            method = "plot_COM"
            sharey = False
            pad = -14
        elif dim == 2:
            projection = "hammer"
            method = "plot_energy_COM"
            sharey = False
            pad = 0
        elif dim == 1:
            projection = None
            method = "plot_energy_violin"
            sharey = True
            pad = 0
        else:
            raise ValueError("Only energy plots with 1, 2 or 3 dimensions possible!")

        self._make_plot_for_all(method, projection=projection, remove_midlabels=False,
                                creation_kwargs={"sharex": False, "sharey": sharey},
                                plotting_kwargs={"projection": projection, "energy_type": energy_type,
                                                 "vmin": vmin, "vmax": vmax})

        titles = [subplot.get_possible_title() for subplot in self.list_plots]
        self.add_titles(list_titles=titles, pad=pad)

        if dim == 3:
            self.unify_axis_limits()
            self.add_colorbar()
        if dim == 2:
            self.add_colorbar()
        if dim == 1:
            self.unify_axis_limits(x_ax=False, y_ax=True)

        if animate_rot and dim == 3:
            self.animate_figure_view(f"energy_{dim}d", dpi=100)
        if save:
            self._save_multiplot(f"energy_{dim}d")

    def create_all_plots(self, and_animations=False, **kwargs):
        # keep this one since we have parameters in function
        self.make_all_COM_3d_plots(animate_rot=and_animations)
        self.make_all_COM_hammer_plots()
        energy_types = self.list_plots[0].parsed_trajectory.energies.labels
        for energy_type in energy_types:
            for dim in (1, 2, 3):
                self.make_all_energy_plots(dim, energy_type=energy_type)