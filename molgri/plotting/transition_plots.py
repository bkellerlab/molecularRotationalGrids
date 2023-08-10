"""
Plots of ITS, eigenvectors and eigenvalues of transition/rate matrix.

A collection of methods to visualise the SqRA or MSM objects.
"""

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
            eigenvals = eigenvals[0]
            for j in range(1, num_eigenv):
                x_min, x_max = self.ax.get_xlim()
                min_value = np.min([0, x_min])
                max_value = np.max([1, x_max])
                self.ax.hlines(- 1 / eigenvals[j], min_value, max_value, color="black", ls="--")

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
    if __name__ == "__main__":
        from molgri.molecules._load_examples import load_molgri_data, load_simulation_data
        from molgri.space.fullgrid import FullGrid
        from molgri.molecules.transitions import MSM, SQRA, SimulationHistogram
        from molgri.plotting.other_plots import ArrayPlot
        from molgri.plotting.molecule_plots import TrajectoryPlot

        USE_SAVED = False

        # TRANSITION MATRIX

        # parsed_sim = load_simulation_data()
        # # define some full grid to assign to
        # full_grid = FullGrid(t_grid_name="linspace(3, 13, 4)", o_grid_name="ico_20", b_grid_name="zero")
        #
        # combined_sim = SimulationHistogram(parsed_sim, full_grid)
        # msm = MSM(combined_sim, energy_type="Potential Energy", use_saved=USE_SAVED)
        # transition_matrix = msm.get_transitions_matrix(
        #
        # )
        # ArrayPlot(transition_matrix[0], default_context="talk").make_heatmap_plot(save=True)
        # tp_msm = TransitionPlot(msm, default_context="talk")
        # tp_msm.make_its_plot(save=True)
        # tp_msm.make_eigenvectors_plot(num_eigenv=3)
        # tp_msm.make_eigenvalues_plot()
        #
        # # RATE MATRIX
        #
        molgri_pt = load_molgri_data()

        tp = TrajectoryPlot(molgri_pt)

        full_grid_m = FullGrid(t_grid_name="linspace(0.8, 1.5, 10)", o_grid_name="ico_50", b_grid_name="zero")
        #
        combined_molgri = SimulationHistogram(molgri_pt, full_grid_m)
        sqra = SQRA(combined_molgri, energy_type="Potential Energy", use_saved=USE_SAVED)
        #rates_matrix = sqra.get_transitions_matrix()
        #ArrayPlot(rates_matrix[0], default_context="talk").make_heatmap_plot(save=True)

        tp_sqra = TransitionPlot(sqra, default_context="talk")
        tp_sqra.make_its_plot(save=True, as_line=True)
        tp_sqra.make_eigenvectors_plot(num_eigenv=3)
        tp_sqra.make_eigenvalues_plot()

        # full_grid = FullGrid(t_grid_name="[5, 15]", o_grid_name="ico_10", b_grid_name="zero")
        # tp = TrajectoryPlot(parsed_sim)
        # ani = tp.make_COM_plot(animate_rot=True, projection="3d", save=True, fg=full_grid)