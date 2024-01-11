"""
Combine a ParsedTrajectory and a FullGrid to generate a MSM or SqRA model.

In this module, the two methods of evaluating transitions between states - the MSM and the SqRA approach - are
implemented.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import eigs
from scipy.constants import k as kB
from scipy.constants import N_A
from tqdm import tqdm

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

    # noinspection PyMissingOrEmptyDocstring
    def get_name(self) -> str:
        traj_name = self.parsed_trajectory.get_name()
        grid_name = self.full_grid.get_name()
        return f"{traj_name}_{grid_name}"

    def get_all_assignments(self) -> Tuple[NDArray, NDArray]:
        """
        For each step in the trajectory assign which cell it belongs to. Uses the default atom selection of the
        ParsedTrajectory object.

        Returns:
            (all centers of mass within grid, all assignments to grid cells)
        """
        atom_selection = self.parsed_trajectory.default_atom_selection
        # if you do nan_free, your tau may not be correct anymore because several steps in-between may be missing
        return self.parsed_trajectory.assign_coms_2_grid_points(self.full_grid, atom_selection=atom_selection,
                                                                nan_free=False)


class TransitionModel(ABC):

    """
    A class that contains both the MSM and SQRA models.
    """

    def __init__(self, sim_hist: SimulationHistogram, tau_array: NDArray = None, use_saved: bool = False):
        self.sim_hist = sim_hist
        if tau_array is None:
            tau_array = np.array([2, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150, 180, 200])
        self.tau_array = tau_array
        self.use_saved = use_saved
        _, self.assignments = self.sim_hist.get_all_assignments()
        self.num_cells = len(self.sim_hist.full_grid.get_full_grid_as_array())
        self.num_tau = len(self.tau_array)
        self.transition_matrix = None

    # noinspection PyMissingOrEmptyDocstring
    def get_name(self) -> str:
        return self.sim_hist.get_name()

    @abstractmethod
    def get_transitions_matrix(self) -> NDArray:
        """For MSM, generate the transition matrix from simulation data. For SQRA, generate rate matrix from the
        point energy calculations.

        Returns:
            an array of shape (num_tau, num_cells, num_cells) for MSM or (1, num_cells, num_cells) for SQRA
        """
        pass

    @save_or_use_saved
    def get_eigenval_eigenvec(self, num_eigenv: int = 15, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Obtain eigenvectors and eigenvalues of the transition matrices.

        Args:
            num_eigenv: how many eigenvalues/vectors pairs to return (too many may give inaccurate results)
            **kwargs: named arguments to forward to eigs()
        Returns:
            (eigenval, eigenvec) a tuple of eigenvalues and eigenvectors, first num_eigv given for all tau-s
            Eigenval is of shape (num_tau, num_columns), eigenvec of shape (num_tau, num_cells, num_columns)
        """
        all_tms = self.get_transitions_matrix()
        all_eigenval = np.zeros((self.num_tau, num_eigenv))
        all_eigenvec = np.zeros((self.num_tau, self.num_cells, num_eigenv))
        for tau_i, tau in enumerate(self.tau_array):
            tm = all_tms[tau_i]  # the transition matrix for this tau
            tm[np.isnan(tm)] = 0  # replace nans with zeros
            # in order to compute left eigenvectors, compute right eigenvectors of the transpose
            eigenval, eigenvec = eigs(tm.T, num_eigenv, maxiter=100000, tol=0, which="LM", sigma=0, **kwargs)
            # don't need to deal with complex outputs in case all values are real
            # TODO: what happens here if we have negative imaginary components?
            if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
                eigenvec = eigenvec.real
                eigenval = eigenval.real
            # sort eigenvectors according to their eigenvalues
            idx = eigenval.argsort()[::-1]
            eigenval = eigenval[idx]
            eigenvec = eigenvec[:, idx]
            all_eigenval[tau_i] = eigenval
            all_eigenvec[tau_i] = eigenvec
        return all_eigenval, all_eigenvec


class MSM(TransitionModel):

    """
    Markov state model (MSM) works on simulation trajectories by discretising the accessed space and counting
    transitions between states in the time span of tau.
    """

    @save_or_use_saved
    def get_transitions_matrix(self, noncorr: bool = False) -> NDArray:
        """
        Obtain a set of transition matrices for different tau-s specified in self.tau_array.

        Args:
            noncorr: bool, should only every tau-th frame be used for MSM construction
                     (if False, use sliding window - much more expensive but throws away less data)
        Returns:
            an array of transition matrices of shape (self.num_tau, self.num_cells, self.num_cells)
        """

        def window(seq: Sequence, len_window: int, step: int = 1) -> Tuple[Any, Any]:
            """
            How this works: returns a list of sublists. Each sublist has two elements: [start_element, end_element].
            Both elements are taken from the seq and are len_window elements apart. The parameter step controls what
            the step between subsequent start_elements is.

            Yields:
                [start_element, end_element] where both elements com from seq where they are len_window positions
                apart

            Example:
                >>> gen_window = window([1, 2, 3, 4, 5, 6], len_window=2, step=3)
                >>> next(gen_window)
                (1, 3)
                >>> next(gen_window)
                (4, 6)
            """
            # in this case always move the window by step and use all points in simulations to count transitions
            for k in range(0, len(seq) - len_window, step):
                start_stop_list = seq[k: k + len_window + 1:len_window]
                if not np.isnan(start_stop_list).any():
                    yield tuple([int(el) for el in start_stop_list])

        def noncorr_window(seq: Sequence, len_window: int) -> Tuple[Any, Any]:
            """
            Subsample the seq so that only each len_window-th element remains and then similarly return pairs of
            elements.

            Example:
                >>> gen_obj = noncorr_window([1, 2, 3, 4, 5, 6, 7], 3)
                >>> next(gen_obj)
                (1, 4)
                >>> next(gen_obj)
                (4, 7)

            """
            # in this case, only use every len_window-th element for MSM. Faster but loses a lot of data
            return window(seq, len_window, step=len_window)

        if self.transition_matrix is None:
            self.transition_matrix = np.zeros(shape=(self.num_tau, self.num_cells, self.num_cells))
            for tau_i, tau in enumerate(tqdm(self.tau_array)):
                # save the number of transitions between cell with index i and cell with index j
                count_per_cell = {(i, j): 0 for i in range(self.num_cells) for j in range(self.num_cells)}
                if not noncorr:
                    window_cell = window(self.assignments, int(tau))
                else:
                    window_cell = noncorr_window(self.assignments, int(tau))
                for cell_slice in window_cell:
                    try:
                        count_per_cell[cell_slice] += 1
                    except KeyError:
                        # the point is outside the grid and assigned to NaN - ignore for now
                        pass
                for key, value in count_per_cell.items():
                    start_cell, end_cell = key
                    self.transition_matrix[tau_i, start_cell, end_cell] += value
                    # enforce detailed balance
                    self.transition_matrix[tau_i, end_cell, start_cell] += value
                # divide each row of each matrix by the sum of that row
                sums = self.transition_matrix[tau_i].sum(axis=-1, keepdims=True)
                sums[sums == 0] = 1
                self.transition_matrix[tau_i] = self.transition_matrix[tau_i] / sums
        return self.transition_matrix


class SQRA(TransitionModel):

    """
    As opposed to MSM, this object works with a pseudo-trajectory that evaluates energy at each grid point and
    with geometric parameters of position space division.
    """

    def __init__(self, sim_hist: SimulationHistogram, energy_type: str = "Potential", **kwargs):
        # SQRA doesn't need a tau array, but for compatibility with MSM we use this one
        tau_array = np.array([1])
        self.energy_type = energy_type
        super().__init__(sim_hist, tau_array=tau_array, **kwargs)
        len_fgrid = len(self.sim_hist.full_grid.get_full_grid_as_array())
        if len(sim_hist.parsed_trajectory.energies.get_energies(energy_type)) == len_fgrid:
            self.assignments = np.arange(len_fgrid)

    @save_or_use_saved
    def get_transitions_matrix(self, D: float = 1, T=273) -> NDArray:
        """
        Return the rate matrix as calculated by the SqRA formula:

        Q_ij = np.sqrt(pi_j/pi_i) * D * S_ij / (h_ij * V_i)

        Args:
            D: diffusion constant
            energy_type: the keyword to pass to self.sim_hist.parsed_trajectory.get_all_energies to obtain energy
            information at centers of cells

        Returns:
            a (1, self.num_cells, self.num_cells) array that is a rate matrix estimated with SqRA (first dimension
            is expanded to be compatible with the MSM model)
        """
        if self.transition_matrix is None:
            all_volumes = self.sim_hist.full_grid.get_total_volumes()
            # TODO: move your work to sparse matrices at some point?
            all_surfaces = self.sim_hist.full_grid.get_full_borders().toarray()
            all_distances = self.sim_hist.full_grid.get_full_distances().toarray()
            # energies are either NaN (calculation in that cell was not completed or the particle left the cell during
            # optimisation) or they are the arithmetic average of all energies of COM assigned to that cell.
            all_energies = np.empty(shape=(self.num_cells,))
            energy_counts = np.zeros(shape=(self.num_cells,))
            obtained_energies = self.sim_hist.parsed_trajectory.get_all_energies(energy_type=self.energy_type)
            for a, e in zip(self.assignments, obtained_energies):
                if not np.isnan(a):
                    all_energies[int(a)] += e
                    energy_counts[int(a)] += 1
            # in both cases avoiding division with zero
            # TODO: instead of averaging find the most central point
            all_energies = np.divide(all_energies, energy_counts, out=np.zeros_like(all_energies),
                                     where=energy_counts != 0)
            self.transition_matrix = np.divide(D * all_surfaces, all_distances, out=np.zeros_like(D * all_surfaces),
                                    where=all_distances!=0)


            for i, _ in enumerate(self.transition_matrix):
                divide_by = all_volumes[i]
                self.transition_matrix[i] = np.divide(self.transition_matrix[i], divide_by,
                                                      out=np.zeros_like(self.transition_matrix[i]),
                                         where=(divide_by != 0 and divide_by != np.NaN))

            for j, _ in enumerate(self.transition_matrix):
                for i, _ in enumerate(self.transition_matrix):
                    # gromacs uses kJ/mol as energy unit, boltzmann constant is J/K
                    multiply_with = np.exp((all_energies[i]-all_energies[j])*1000/(2*kB*N_A*T))
                    self.transition_matrix[i, j] = np.multiply(self.transition_matrix[i, j], multiply_with,
                                                               out=np.zeros_like(self.transition_matrix[i, j]),
                                               where=multiply_with != np.NaN)

            # normalise rows
            sums = np.sum(self.transition_matrix, axis=1)
            np.fill_diagonal(self.transition_matrix, -sums)
            # additional axis
            self.transition_matrix = self.transition_matrix[np.newaxis, :]
            assert self.transition_matrix.shape == (1, self.num_cells, self.num_cells)
        return self.transition_matrix


if __name__ == "__main__":
    import pandas as pd
    from molgri.molecules.parsers import FileParser, ParsedEnergy, XVGParser
    import MDAnalysis as mda
    from molgri.space.utils import k_argmin_in_array

    my_num = "0099"

    # preparing the parsed trajectory
    my_parser = XVGParser(f"/home/hanaz63/nobackup/gromacs/H2O_H2O_{my_num}/H2O_H2O_{my_num}.xvg")
    pe = my_parser.get_parsed_energy()  # .get_energies("Disper. corr.")
    pt_parser = FileParser(
        path_topology=f"/home/hanaz63/nobackup/gromacs/H2O_H2O_{my_num}/H2O_H2O_{my_num}.gro",
        path_trajectory=f"/home/hanaz63/nobackup/gromacs/H2O_H2O_{my_num}/H2O_H2O_{my_num}.xtc")
    parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="bynum 4:6")
    parsed_trajectory.energies = pe

    energ = pe.get_energies("Potential")


    # if the len of pt == len of fg then you can just assign 1 to 1

    fg = FullGrid(o_grid_name="12", b_grid_name="8", t_grid_name="linspace(0.2, 0.5, 10)", use_saved=True)
    #fg = FullGrid(o_grid_name="42", b_grid_name="40", t_grid_name="linspace(0.2, 0.5, 20)", use_saved=True)
    print(len(energ), len(fg.get_full_grid_as_array()))

    sim_hist = SimulationHistogram(parsed_trajectory, full_grid=fg)
    #print(sim_hist.get_all_assignments())

    u = mda.Universe(f"/home/hanaz63/nobackup/gromacs/H2O_H2O_{my_num}/H2O_H2O_{my_num}.gro",
                     f"/home/hanaz63/nobackup/gromacs/H2O_H2O_{my_num}/H2O_H2O_{my_num}.xtc")
    sqra = SQRA(sim_hist)
    my_rates = sqra.get_transitions_matrix()