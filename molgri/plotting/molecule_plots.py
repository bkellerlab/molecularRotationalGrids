import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns

from molgri.constants import ENERGY2SHORT
from molgri.molecules.parsers import ParsedTrajectory
from molgri.plotting.abstract import RepresentationCollection, MultiRepresentationCollection


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
        self.ax.scatter(*data, **kwargs)
        if projection == "3d":
            self._equalize_axes()
        elif projection == "hammer":
            self.ax.set_xticks([])

    def get_possible_title(self):
        return f"N={self.N_used}"

    def make_COM_plot(self, ax=None, fig=None, save=True, atom_selection=None, projection="3d", animate_rot=False):
        self._create_fig_ax(ax=ax, fig=fig, projection=projection)
        # if a pseudotrajectory, default setting is to plot COM or r_molecule
        atom_selection = self._default_atom_selection(atom_selection)

        # filter out unique COMs
        coms, _ = self.parsed_trajectory.get_unique_com_till_N(N=self.N_used, atom_selection=atom_selection)

        # plot data
        self._make_scatter_plot(projection, coms.T, color="black")

        if save:
            self._save_plot_type(f"com_{projection}")

        if animate_rot and projection == "3d":
            self._animate_figure_view(self.fig, self.ax, f"com_rotated")

    def make_energy_COM_plot(self, ax=None, fig=None, save=True, atom_selection=None, projection="3d",
                             animate_rot=False, energy_type="Potential"):
        self._create_fig_ax(ax=ax, fig=fig, projection=projection)
        atom_selection = self._default_atom_selection(atom_selection)

        coms, energies = self.parsed_trajectory.get_unique_com_till_N(N=self.N_used, energy_type=energy_type,
                                                                      atom_selection=atom_selection)

        # convert energies to colors
        cmap = cm.coolwarm
        norm = Normalize(vmin=np.min(energies), vmax=np.max(energies))

        # plot data
        self._make_scatter_plot(projection, coms.T, cmap=cmap, c=energies, norm=norm)

        save_name = f"energies_{ENERGY2SHORT[energy_type]}"
        if save:
            self._save_plot_type(f"{save_name}_{projection}")

        if animate_rot and projection == "3d":
            self._animate_figure_view(self.fig, self.ax, f"{save_name}_rotated")

    def make_molecule_plot(self):
        pass

    # kwargs are strictly necessary!
    def make_energy_violin_plot(self, ax=None, fig=None, save=True, energy_type: str = "Potential", **kwargs):
        self._create_fig_ax(ax=ax, fig=fig)
        coms, energies = self.parsed_trajectory.get_unique_com_till_N(N=self.N_used, energy_type=energy_type)
        sns.violinplot(energies, ax=self.ax, scale="count", inner="stick", cut=0)
        self.ax.set_xticklabels([])
        #self.ax.set_xlabel("N")
        self.ax.set_ylabel(f"{energy_type} [kJ/mol]")

        save_name = f"violin_energies_{ENERGY2SHORT[energy_type]}"
        if save:
            self._save_plot_type(save_name)

    def create_all_plots(self, and_animations=False):
        self.make_COM_plot(projection="3d", animate_rot=and_animations)
        self.make_COM_plot(projection="hammer")
        for energy_type in self.parsed_trajectory.energies.labels:
            self.make_energy_COM_plot(projection="3d", animate_rot=and_animations, energy_type=energy_type)
            self.make_energy_COM_plot(projection="hammer", energy_type=energy_type)
            self.make_energy_violin_plot(energy_type=energy_type)


class ConvergenceMultiCollection(MultiRepresentationCollection):

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
        self._make_plot_for_all("make_COM_plot", projection="3d", remove_midlabels=False,
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
        self._make_plot_for_all("make_COM_plot", projection="hammer", remove_midlabels=False,
                                creation_kwargs={"sharex": False, "sharey": False},
                                plotting_kwargs={"projection": "hammer"})

        titles = [subplot.get_possible_title() for subplot in self.list_plots]
        self.add_titles(list_titles=titles)

        if save:
            self._save_multiplot(f"COM_hammer")

    def make_all_energy_plots(self, dim, animate_rot=False, save=True, energy_type: str = "Potential"):
        if dim == 3:
            projection = "3d"
            method = "make_energy_COM_plot"
            sharey = False
            pad = -14
        elif dim == 2:
            projection = "hammer"
            method = "make_energy_COM_plot"
            sharey = False
            pad = 0
        elif dim == 1:
            projection = None
            method = "make_energy_violin_plot"
            sharey = True
            pad = 0
        else:
            raise ValueError("Only energy plots with 1, 2 or 3 dimensions possible!")

        self._make_plot_for_all(method, projection=projection, remove_midlabels=False,
                                creation_kwargs={"sharex": False, "sharey": sharey},
                                plotting_kwargs={"projection": projection, "energy_type": energy_type})

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
            self.animate_figure_view(f"energy_{dim}d")
        if save:
            self._save_multiplot(f"energy_{dim}d")

    def create_all_plots(self, and_animations=False):
        self.make_all_COM_3d_plots(animate_rot=and_animations)
        self.make_all_COM_hammer_plots()
        energy_types = self.list_plots[0].parsed_trajectory.energies.labels
        for energy_type in energy_types:
            for dim in (1, 2, 3):
                self.make_all_energy_plots(dim, energy_type=energy_type)
