import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from molgri.plotting.abstract import RepresentationCollection
from molgri.molecules.transitions import TransitionModel
from molgri.plotting.fullgrid_plots import FullGridPlot
from molgri.constants import DIM_SQUARE


class TransitionPlot(RepresentationCollection):

    def __init__(self, transition_obj: TransitionModel, *args, **kwargs):
        self.transition_obj = transition_obj
        data_name = self.transition_obj.get_name()
        super().__init__(data_name, *args, **kwargs)

    def make_its_plot(self, fig=None, ax=None, save=True, num_eigenv=6, as_line=False):
        """
        Plot iterative timescales.
        """
        self._create_fig_ax(ax=ax, fig=fig)
        tau_array = self.transition_obj.tau_array
        eigenvals, eigenvecs = self.transition_obj.get_eigenval_eigenvec(num_eigenv=num_eigenv)
        eigenvals = np.array(eigenvals)
        dt = self.transition_obj.sim_hist.parsed_trajectory.dt

        if not as_line:
            for j in range(1, num_eigenv):
                to_plot_abs = np.array(-tau_array * dt / np.log(np.abs(eigenvals[:, j])))
                sns.lineplot(x=tau_array * dt, y=to_plot_abs,
                             ax=ax, legend=False)
        else:
            # for SQRA plot vertical lines
            tau_array = np.concatenate(np.array([0]), tau_array)
            for j in range(1, len(eigenvals[:num_eigenv])):
                absolute_its = np.array([- 1 / eigenvals[j] for _ in tau_array])
                ax.plot(tau_array * dt, absolute_its, color="black", ls="--")

        self.ax.set_xlim(left=0, right=tau_array[-1] * dt)
        self.ax.set_xlabel(r"$\tau$")
        self.ax.set_ylabel(r"ITS")
        tau_array_with_zero = (tau_array * dt).tolist()
        tau_array_with_zero.append(0)
        tau_array_with_zero.sort()
        self.ax.fill_between(tau_array_with_zero, tau_array_with_zero, color="grey", alpha=0.5)

        if save:
            self._save_plot_type(f"its")

    def make_eigenvalues_plot(self, fig=None, ax=None, save=True, num_eigenv=None):
        """
        Visualize the eigenvalues of rate matrix.
        """

        self._create_fig_ax(ax=ax, fig=fig)

        eigenvals, _ = self.transition_obj.get_eigenval_eigenvec()
        eigenvals = np.array(eigenvals)[0]

        if num_eigenv:
            eigenvals = eigenvals[:num_eigenv]

        xs = np.linspace(0, 1, num=len(eigenvals))
        self.ax.scatter(xs, eigenvals, s=5, c="black")
        for i, eigenw in enumerate(eigenvals):
            self.ax.vlines(xs[i], eigenw, 0, linewidth=0.5, color="black")
        self.ax.hlines(0, 0, 1, color="black")
        self.ax.set_ylabel(f"Eigenvalues")
        self.ax.axes.get_xaxis().set_visible(False)

        if save:
            self._save_plot_type(f"eigenvalues")

    def make_eigenvectors_plot(self, ax=None, fig=None, save=True, num_eigenv: int = 5, projection="3d"):
        """
        Visualize the energy surface and the first num (default=3) eigenvectors
        """

        self.fig, self.ax = plt.subplots(1, num_eigenv, subplot_kw={"projection": projection},
                                         figsize=(DIM_SQUARE[0], num_eigenv*DIM_SQUARE[0]))

        for i, subax in enumerate(self.ax.ravel()):
            self.make_one_eigenvector_plot(i, ax=subax, fig=self.fig, projection=projection, save=False)

        if save:
            self._save_plot_type(f"eigenvectors_{projection}")

    def make_one_eigenvector_plot(self, eigenvec_index: int, ax=None, fig=None, save=True, projection="3d",
                                  animate_rot=False):
        self._create_fig_ax(ax=ax, fig=fig, projection=projection)

        eigenvals, eigenvecs = self.transition_obj.get_eigenval_eigenvec()

        # shape: (number_taus, number_cells, num_eigenvectors)
        eigenvecs = eigenvecs[0]  # values for the first tau
        eigenvecs = eigenvecs.T

        fgp = FullGridPlot(self.transition_obj.sim_hist.full_grid, default_complexity_level="half_empty")
        if projection == "3d":
            fgp.make_full_voronoi_plot(ax=self.ax, fig=self.fig, plot_vertex_points=False, save=False)
        fgp.make_position_plot(ax=self.ax, fig=self.fig, save=False, c=eigenvecs[eigenvec_index], animate_rot=False,
                               projection=projection)
        self.ax.set_title(f"Eigenv. {eigenvec_index}")
        if save:
            self._save_plot_type(f"eigenvector_{eigenvec_index}_{projection}")
        if animate_rot:
            return self._animate_figure_view(self.fig, self.ax, f"eigenvector_{eigenvec_index}_rotated")


if __name__ == "__main__":
    import pandas as pd
    from molgri.molecules.parsers import FileParser, ParsedEnergy
    from molgri.space.fullgrid import FullGrid
    from molgri.molecules.transitions import SimulationHistogram
    from molgri.molecules.transitions import MSM, SQRA

    # SIMULATION + MSM MODEL
    # preparing the parsed trajectory
    # pt_parser = FileParser(
    #     path_topology="/home/mdglasius/Modelling/trypsin_normal/inputs/trypsin_probe.pdb",
    #     path_trajectory="/home/mdglasius/Modelling/trypsin_normal/nobackup/outputs/aligned_traj.dcd")
    # parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="segid B")
    #
    # # preparing the grid
    # fg = FullGrid(t_grid_name="[5, 10, 15]", o_grid_name="ico_10", b_grid_name="zero")
    #
    # sh = SimulationHistogram(parsed_trajectory, fg)
    # my_msm = MSM(sh, use_saved=False)
    #
    # tp = TransitionPlot(my_msm)
    # tp.make_its_plot()
    # tp.make_eigenvalues_plot()
    #
    # tp.make_eigenvectors_plot(projection="3d")
    # tp.make_eigenvectors_plot(projection="hammer")

    # MOLGRI + SQRA
    # preparing the parsed trajectory
    path_energy = "/home/mdglasius/Modelling/trypsin_test/output/measurements/temp_minimized.csv"
    df = pd.read_csv(path_energy)
    energies = df['potential'].to_numpy()[:, np.newaxis]
    pe = ParsedEnergy(energies=energies, labels=["Potential"], unit="(kJ/mole)")
    pt_parser = FileParser(
        path_topology="/home/mdglasius/Modelling/trypsin_test/output/pt_files/final_trypsin_NH4_o_ico_512_b_zero_1_t_2153868773.gro",
        path_trajectory="/home/mdglasius/Modelling/trypsin_test/output/pt_files/minimizedPT.pdb")
    parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="not protein")
    parsed_trajectory.energies = pe

    # the exact grid used
    fg = FullGrid(t_grid_name="linspace(0.8, 2.5, 2)", o_grid_name="ico_80", b_grid_name="zero")

    # plotting
    sh = SimulationHistogram(parsed_trajectory, fg)
    my_sqra = SQRA(sh, use_saved=False)

    tp = TransitionPlot(my_sqra)
    tp.make_its_plot(as_line=True)
    tp.make_eigenvalues_plot()
    for i in range(3):
        tp.make_one_eigenvector_plot(i, animate_rot=True)

    tp.make_eigenvectors_plot(projection="3d")
    tp.make_eigenvectors_plot(projection="hammer")