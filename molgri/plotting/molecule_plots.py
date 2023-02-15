import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns

from molgri.constants import ENERGY2SHORT, ENERGY_NO_UNIT
from molgri.molecules.parsers import ParsedTrajectory, get_unique_com, PtParser
from molgri.plotting.abstract import RepresentationCollection, MultiRepresentationCollection


class TrajectoryPlot(RepresentationCollection):

    def __init__(self, parsed_trajectory: ParsedTrajectory):
        self.parsed_trajectory = parsed_trajectory
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

    def make_COM_plot(self, ax=None, fig=None, save=True, atom_selection=None, projection="3d", animate_rot=False):
        self._create_fig_ax(ax=ax, fig=fig, projection=projection)
        # if a pseudotrajectory, default setting is to plot COM or r_molecule
        atom_selection = self._default_atom_selection(atom_selection)

        # get data
        coms = self.parsed_trajectory.get_all_COM(atom_selection)
        # filter out unique ones
        coms, _ = get_unique_com(coms, None)

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

        # get energies
        energies = self.parsed_trajectory.get_all_energies(energy_type)
        if energies is None:
            print(f"This energy_COM plot could not be plotted since energies of {self.data_name} were not found.")
            return
        coms = self.parsed_trajectory.get_all_COM(atom_selection)
        # filter for lowest energy per point
        coms, energies = get_unique_com(coms, energies)

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

    def make_voranoi_cells_plot(self, ax=None, fig=None, save=True):
        points = self.parsed_trajectory.get_all_COM(atom_selection=None)
        svs = voranoi_surfaces_on_stacked_spheres(points)
        for i, sv in enumerate(svs):
            sv.sort_vertices_of_regions()
            t_vals = np.linspace(0, 1, 2000)
            # plot Voronoi vertices
            self.ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')
            # indicate Voronoi regions (as Euclidean polygons)
            for region in sv.regions:
                n = len(region)
                for j in range(n):
                    start = sv.vertices[region][j]
                    end = sv.vertices[region][(j + 1) % n]
                    norm = np.linalg.norm(start)
                    result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)
                    self.ax.plot(norm * result[..., 0], norm * result[..., 1], norm * result[..., 2], c='k')

    def make_molecule_plot(self):
        pass

    def make_convergence_plot(self, method_name, **method_args):
        pass

    def make_energy_violin_plot(self, ax=None, fig=None, save=True, energy_type: str = "Potential"):
        self._create_fig_ax(ax=ax, fig=fig)
        energies = self.parsed_trajectory.get_all_energies(energy_type)
        sns.violinplot(energies, ax=self.ax, scale="count", inner="stick", cut=0)
        self.ax.set_xticklabels([str(len(energies))])
        self.ax.set_xlabel("N")
        self.ax.set_ylabel(f"{energy_type} [kJ/mol]")

        save_name = f"violin_energies_{ENERGY2SHORT[energy_type]}"
        if save:
            self._save_plot_type(save_name)

    def create_all_plots(self, and_animations=False):
        self.make_COM_plot(projection="3d", animate_rot=and_animations)
        self.make_COM_plot(projection="hammer")
        for energy_type in ENERGY_NO_UNIT:
            self.make_energy_COM_plot(projection="3d", animate_rot=and_animations, energy_type=energy_type)
            self.make_energy_COM_plot(projection="hammer", energy_type=energy_type)
            self.make_energy_violin_plot(energy_type=energy_type)


class ConvergenceMultiCollection(MultiRepresentationCollection):

    def __init__(self, parsed_trajectory: ParsedTrajectory, N_set: tuple = None):
        data_name = f"convergence_{parsed_trajectory.get_name()}"
        # TODO: generate parsed_trajectories with limited number of points
        #list_plots = [TrajectoryPlot()]
       # super().__init__(data_name, list_plots)


if __name__ == "__main__":
    import os

    topology_path = os.path.join("output", "pt_files", "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.gro")
    trajectory_path = os.path.join("output", "pt_files", "H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xtc")
    water_path = os.path.join("input", "base_gro_files", "H2O.gro")
    parser_pt = PtParser(m1_path=water_path, m2_path=water_path, path_topology=topology_path, path_trajectory=trajectory_path).get_parsed_trajectory()
    TrajectoryPlot(parser_pt).create_all_plots()
