"""
In this module, the two methods of evaluating transitions between states - the MSM and the SqRA approach - are
implemented.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import eigs
from tqdm import tqdm

from molgri.paths import PATH_OUTPUT_PLOTS
from molgri.wrappers import save_or_use_saved
from molgri.molecules.parsers import ParsedTrajectory
from molgri.space.fullgrid import FullGrid


class SimulationHistogram:

    """
    This class takes the data from ParsedTrajectory and combines it with a specific FullGrid so that each simulation
    step can be assigned to a cell and the occupancy of those cells evaluated.
    """

    def __init__(self, parsed_trajectory: ParsedTrajectory, full_grid: FullGrid):
        self.parsed_trajectory = parsed_trajectory
        self.full_grid = full_grid

    def get_name(self):
        traj_name = self.parsed_trajectory.get_name()
        grid_name = self.full_grid.get_name()
        return f"{traj_name}_{grid_name}"

    def get_all_assignments(self):
        """
        For each step in the trajectory assign which cell it belongs to.
        """
        atom_selection = self.parsed_trajectory.default_atom_selection
        return self.parsed_trajectory.assign_coms_2_grid_points(self.full_grid, atom_selection=atom_selection)


class MSM:

    """
    Markov state model (MSM) works on simulation trajectories by discretising the accessed space and counting
    transitions between states in the time span of tau.
    """

    def __init__(self, sim_hist: SimulationHistogram, tau_array: NDArray = None, use_saved: bool = False):
        self.sim_hist = sim_hist
        if tau_array is None:
            tau_array = np.array([2, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150, 180, 200])
        self.tau_array = tau_array
        self.use_saved = use_saved
        self.assignments = self.sim_hist.get_all_assignments()
        self.num_cells = len(self.sim_hist.full_grid.get_flat_position_grid())

    def get_name(self):
        return self.sim_hist.get_name()

    @save_or_use_saved
    def get_transitions_matrix(self, tau_array: np.ndarray = None, noncorr: bool = False):
        """
        Obtain a set of transition matrices for different tau-s specified in tau_array.

        Args:
            tau_array: 1D array of tau values for which the transition matrices should be constructed
            noncorr: bool, should only every tau-th frame be used for MSM construction
                     (if False, use sliding window - much more expensive but throws away less data)
        Returns:
            an array of transition matrices
        """
        if tau_array:
            self.tau_array = tau_array

        def window(seq, len_window, step=1):
            # in this case always move the window by step and use all points in simulations to count transitions
            return [seq[k: k + len_window:len_window-1] for k in range(0, (len(seq)+1)-len_window, step)]

        def noncorr_window(seq, len_window):
            # in this case, only use every tau-th element for MSM. Faster but loses a lot of data
            cut_seq = seq[0:-1:len_window]
            return [[a, b] for a, b in zip(cut_seq[0:-2], cut_seq[1:])]

        all_matrices = []
        for tau_i, tau in enumerate(tqdm(self.tau_array)):
            transition_matrix = np.zeros(shape=(self.num_cells, self.num_cells))
            count_per_cell = {(i, j): 0 for i in range(self.num_cells) for j in range(self.num_cells)}
            if not noncorr:
                window_cell = window(self.assignments, int(tau))
            else:
                window_cell = noncorr_window(self.assignments, int(tau))
            for cell_slice in window_cell:
                start_cell = cell_slice[0]
                end_cell = cell_slice[1]
                try:
                    count_per_cell[(start_cell, end_cell)] += 1
                except KeyError:
                    pass
            for key, value in count_per_cell.items():
                start_cell, end_cell = key
                transition_matrix[start_cell, end_cell] += value
                # enforce detailed balance
                transition_matrix[end_cell, start_cell] += value
            # divide each row of each matrix by the sum of that row
            sums = transition_matrix.sum(axis=-1, keepdims=True)
            sums[sums == 0] = 1
            transition_matrix = transition_matrix / sums
            all_matrices.append(transition_matrix)
        return np.array(all_matrices)

    @save_or_use_saved
    def get_eigenval_eigenvec(self, num_eigenv: int = 15, **kwargs):
        """
        Obtain eigenvectors and eigenvalues of the transition matrices.

        Args:
            num_eigv: how many eigenvalues/vectors pairs
            **kwargs: named arguments to forward to eigs()
        Returns:
            (eigenval, eigenvec) a tuple of eigenvalues and eigenvectors, first num_eigv given for all tau-s
        """
        all_tms = self.get_transitions_matrix()
        all_eigenval = []
        all_eigenvec =[]
        for tau_i, tau in enumerate(self.tau_array):
            tm = all_tms[tau_i]
            tm = tm.T
            eigenval, eigenvec = eigs(tm, num_eigenv, maxiter=100000, tol=0, **kwargs)
            if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
                eigenvec = eigenvec.real
                eigenval = eigenval.real
            # sort eigenvectors according to their eigenvalues
            idx = eigenval.argsort()[::-1]
            eigenval = eigenval[idx]
            eigenvec = eigenvec[:, idx]
            all_eigenval.append(eigenval)
            all_eigenvec.append(eigenvec)
        return np.array(all_eigenval), np.array(all_eigenvec)


class SQRA:

    """
    As opposed to MSM, this object works with a pseudo-trajectory that evaluates energy at each grid point and
    with geometric parameters of position space division.
    """


if __name__ == "__main__":
    import pandas as pd
    from molgri.molecules.parsers import FileParser, ParsedEnergy

    # preparing the parsed trajectory
    path_energy = "/home/mdglasius/Modelling/trypsin_normal/nobackup/outputs/data.txt"
    df = pd.read_csv(path_energy)
    energies = df['Potential Energy (kJ/mole)'].to_numpy()[:, np.newaxis]
    pe = ParsedEnergy(energies=energies, labels=["Potential Energy"], unit="(kJ/mole)")
    pt_parser = FileParser(
        path_topology="/home/mdglasius/Modelling/trypsin_normal/inputs/trypsin_probe.pdb",
        path_trajectory="/home/mdglasius/Modelling/trypsin_normal/nobackup/outputs/aligned_traj.dcd")
    parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="segid B")
    parsed_trajectory.energies = pe

    # preparing the grid
    fg = FullGrid(t_grid_name="[5, 10, 15]", o_grid_name="ico_100", b_grid_name="zero")

    sh = SimulationHistogram(parsed_trajectory, fg)
    my_msm = MSM(sh, tau_array=np.array([10, 50, 100]), use_saved=False)