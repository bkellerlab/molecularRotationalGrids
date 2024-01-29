"""
Plots of ITS, eigenvectors and eigenvalues of transition/rate matrix.

A collection of methods to visualise the SqRA or MSM objects.
"""

import numpy as np
import seaborn as sns
import matplotlib.colors as colors

from molgri.plotting.abstract import RepresentationCollection
from molgri.molecules.transitions import TransitionModel, SimulationHistogram
from molgri.plotting.fullgrid_plots import FullGridPlot
from molgri.wrappers import plot3D_method, plot_method


class TransitionPlot(RepresentationCollection):

    def __init__(self, transition_obj: SimulationHistogram, tau_array=None, *args, **kwargs):
        self.simulation_histogram = transition_obj
        self.transition_obj = transition_obj.get_transition_model(tau_array=tau_array)
        data_name = self.transition_obj.get_name()
        super().__init__(data_name, *args, **kwargs)

    @plot_method
    def plot_heatmap(self, trans_index: int = 0):
        """
        This method draws the array and colors the fields according to their values (red = very large,
        blue = very small). Zero values are always white, negative ones always blue, positive ones always red.
        """
        transition_matrix = self.transition_obj.get_transitions_matrix()[trans_index]
        if np.all(transition_matrix< 0):
            cmap = "Blues"
            norm = None
        elif np.all(transition_matrix > 0):
            cmap = "Reds"
            norm = None
        else:
            cmap = "bwr"
            norm = colors.TwoSlopeNorm(vcenter=0, vmax=5, vmin=-5)
        sns.heatmap(transition_matrix, cmap=cmap, ax=self.ax, xticklabels=False, yticklabels=False, norm=norm)
        self._equalize_axes()

    @plot_method
    def plot_its(self, num_eigenv=6, as_line=False, dt=1):
        """
        Plot iterative timescales.
        """

        tau_array = self.transition_obj.tau_array
        eigenvals, eigenvecs = self.transition_obj.get_eigenval_eigenvec(num_eigenv=num_eigenv)
        eigenvals = np.array(eigenvals)
        dt = dt

        if not as_line:
            for j in range(1, num_eigenv):
                to_plot_abs = np.array(-tau_array * dt / np.log(np.abs(eigenvals[:, j])))
                sns.lineplot(x=tau_array * dt, y=to_plot_abs,
                             ax=self.ax, legend=False)
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

    
    @plot_method
    def plot_eigenvalues(self, num_eigenv=None, index_tau=0):
        """
        Visualize the eigenvalues of rate matrix.
        """

        eigenvals, _ = self.transition_obj.get_eigenval_eigenvec(num_eigenv=num_eigenv)
        eigenvals = np.array(eigenvals)[index_tau]

        xs = np.linspace(0, 1, num=len(eigenvals))
        self.ax.scatter(xs, eigenvals, s=5, c="black")
        for i, eigenw in enumerate(eigenvals):
            self.ax.vlines(xs[i], eigenw, 0, linewidth=0.5, color="black")
        self.ax.hlines(0, 0, 1, color="black")
        self.ax.set_ylabel(f"Eigenvalues")
        self.ax.axes.get_xaxis().set_visible(False)

    def plot_eigenvectors(self, num_eigenvectors: int = 5, projection="3d"):
        """
        Visualize the energy surface and the first num (default=5) eigenvectors
        """
        self._create_fig_ax(num_columns=num_eigenvectors, projection=projection)
        for i, subax in enumerate(self.ax.ravel()):
            self.plot_one_eigenvector(i, ax=subax, fig=self.fig, projection=projection, save=False)

    @plot3D_method
    def plot_one_eigenvector(self, eigenvec_index: int = 1):
        eigenvals, eigenvecs = self.transition_obj.get_eigenval_eigenvec()

        # shape: (number_taus, number_cells, num_eigenvectors)
        eigenvecs = eigenvecs[0]  # values for the first tau
        eigenvecs = eigenvecs.T

        fgp = FullGridPlot(self.simulation_histogram.full_grid, default_complexity_level="half_empty")
        fgp.plot_position_voronoi(ax=self.ax, fig=self.fig, plot_vertex_points=False, save=False)
        fgp.plot_positions(ax=self.ax, fig=self.fig, save=False, animate_rot=False) #, c=eigenvecs[eigenvec_index]
        self.ax.set_title(f"Eigenv. {eigenvec_index}")

    @plot_method
    def plot_one_eigenvector_flat(self, eigenvec_index: int = 1, index_tau=0):
        eigenvals, eigenvecs = self.transition_obj.get_eigenval_eigenvec()

        # shape: (number_taus, number_cells, num_eigenvectors)
        eigenvecs = eigenvecs[index_tau]  # values for the first tau
        sns.lineplot(eigenvecs.T[eigenvec_index], ax=self.ax)
        self.ax.set_title(f"Eigenv. {eigenvec_index}")


if __name__ == "__main__":
    pass