import numpy as np
import seaborn as sns

from molgri.plotting.abstract import RepresentationCollection
from molgri.molecules.transitions import MSM


class TransitionPlot(RepresentationCollection):

    def __init__(self, transition_obj: MSM, *args, **kwargs):
        self.transition_obj = transition_obj
        data_name = self.transition_obj.get_name()
        super().__init__(data_name, *args, **kwargs)

    def make_its_plot(self, fig=None, ax=None, save=True, num_eigenv=6):
        """
        Plot iterative timescales.
        """
        self._create_fig_ax(ax=ax, fig=fig)
        tau_array = self.transition_obj.tau_array
        eigenvals, eigenvecs = self.transition_obj.get_eigenval_eigenvec(num_eigenv=num_eigenv)
        eigenvals = np.array(eigenvals)
        dt = self.transition_obj.sim_hist.parsed_trajectory.dt

        for j in range(1, num_eigenv):
            to_plot_abs = np.array(-tau_array * dt / np.log(np.abs(eigenvals[:, j])))
            sns.lineplot(x=tau_array * dt, y=to_plot_abs,
                         ax=ax, legend=False)

        self.ax.set_xlim(left=0, right=tau_array[-1] * dt)
        self.ax.set_xlabel(r"$\tau$")
        self.ax.set_ylabel(r"ITS")
        tau_array_with_zero = (tau_array * dt).tolist()
        tau_array_with_zero.append(0)
        tau_array_with_zero.sort()
        self.ax.fill_between(tau_array_with_zero, tau_array_with_zero, color="grey", alpha=0.5)

        if save:
            self._save_plot_type(f"its_{self.data_name}")


if __name__ == "__main__":
    from molgri.molecules.parsers import FileParser
    from molgri.space.fullgrid import FullGrid
    from molgri.molecules.transitions import SimulationHistogram

    # preparing the parsed trajectory
    pt_parser = FileParser(
        path_topology="/home/mdglasius/Modelling/trypsin_normal/inputs/trypsin_probe.pdb",
        path_trajectory="/home/mdglasius/Modelling/trypsin_normal/nobackup/outputs/aligned_traj.dcd")
    parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="segid B")

    # preparing the grid
    fg = FullGrid(t_grid_name="[5, 10, 15]", o_grid_name="ico_100", b_grid_name="zero")

    sh = SimulationHistogram(parsed_trajectory, fg)
    my_msm = MSM(sh, use_saved=True)

    TransitionPlot(my_msm).make_its_plot()