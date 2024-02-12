"""
Combine a ParsedTrajectory and a FullGrid to generate a MSM or SqRA model.

In this module, the two methods of evaluating transitions between states - the MSM and the SqRA approach - are
implemented.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Any

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from numpy.typing import NDArray
from scipy.sparse import csr_array, diags
from scipy.sparse.linalg import eigs
from scipy.sparse import dok_array
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.constants import k as kB, N_A
from tqdm import tqdm

from molgri.molecules.writers import PtIOManager
from molgri.space.translations import get_between_radii
from molgri.space.voronoi import in_hull
from molgri.wrappers import save_or_use_saved
from molgri.molecules.parsers import ParsedMolecule, XVGParser, FileParser
from molgri.space.fullgrid import FullGrid
from molgri.paths import PATH_OUTPUT_AUTOSAVE, PATH_OUTPUT_ENERGIES, PATH_OUTPUT_LOGGING, PATH_OUTPUT_PT, \
    PATH_INPUT_BASEGRO
from molgri.space.utils import angle_between_vectors, dist_on_sphere, distance_between_quaternions, \
    hemisphere_quaternion_set, k_argmin_in_array, \
    norm_per_axis, \
    normalise_vectors, q_in_upper_sphere


def determine_positive_directions(current_universe, second_molecule):
    pas = current_universe.select_atoms(second_molecule).principal_axes()
    com = current_universe.select_atoms(second_molecule).center_of_mass()
    directions = [0, 0, 0]
    for atom_pos in current_universe.select_atoms(second_molecule).positions:
        for i, pa in enumerate(pas):
            # need to round to avoid problems - assigning direction with atoms very close to 0
            cosalpha = np.round(pa.dot(atom_pos-com), 6)
            directions[i] = np.sign(cosalpha)
        if not np.any(np.isclose(directions,0)):
            break
    # TODO: if exactly one unknown use the other two and properties of righthanded systems to get third
    if np.sum(np.isclose(directions,0)) == 1:
        # only these combinations of directions are possible in righthanded coordinate systems
        allowed_righthanded = [[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]]
        for ar in allowed_righthanded:
            # exactly two identical (and the third is zero)
            if np.sum(np.isclose(ar, directions)) == 2:
                directions = ar
                break
    # if two (or three - that would just be an atom) unknowns raise an error
    elif np.sum(np.isclose(directions,0)) > 1:
        raise ValueError("All atoms perpendicular to at least one of principal axes, can´t determine direction.")
    return np.array(directions)

class SimulationHistogram:

    """
    This class takes the data from ParsedTrajectory and combines it with a specific FullGrid so that each simulation
    step can be assigned to a cell and the occupancy of those cells evaluated.
    """

    def __init__(self, trajectory_name: str, reference_name: str, is_pt: bool, second_molecule_selection: str, full_grid: FullGrid = None,
                 use_saved=True):
        """
        Situation: you have created a (pseudo)trajectory (.gro and .xtc files in PATH_OUTPUT_PT) and calculated the
        energies for every frame (.xvg file PATH_OUTPUT_ENERGIES) - all files have the name given by trajectory_name.
        Now you want to assign one of the molecules (selected by second_molecule_selection) in every frame to one
        cell ub full_grid.

        Args:
            trajectory_name (str): like H2O_H2O_0095 for PTs or H2O_H2O_0095_1000 for trajectories, the folder
                                   PATH_OUTPUT_PT will be searched for this name
            full_grid (FullGrid): used to assign to cells; even if the input is a PT, you can use a different FullGrid
                                  but if None, the FullGrid used in creation will be used
            second_molecule_selection (str): will be forwarded to trajectory_universe to identify the moving molecule

        """
        self.trajectory_name = trajectory_name
        split_name = self.trajectory_name.strip().split("_")
        self.m1_name = split_name[0]
        self.m2_name = split_name[1]
        self.is_pt = is_pt
        self.use_saved = use_saved
        self.second_molecule_selection = second_molecule_selection
        self.energies = None
        self.trajectory_universe = mda.Universe(f"{PATH_OUTPUT_PT}{trajectory_name}.gro",
                                                f"{PATH_OUTPUT_PT}{trajectory_name}.trr")
        try:
            self.reference_universe = mda.Universe(f"{PATH_INPUT_BASEGRO}{reference_name}.gro")
        except:
            self.reference_universe = mda.Universe(f"{PATH_INPUT_BASEGRO}{reference_name}.xyz")
        if full_grid is None and self.is_pt:
            full_grid = self.read_fg_PT()
        self.full_grid = full_grid

        # assignments
        self.position_assignments = None
        self.quaternion_assignments = None
        self.full_assignments = None
        self.transition_model = None

    def __len__(self):
        return len(self.trajectory_universe.trajectory)

    # noinspection PyMissingOrEmptyDocstring
    def get_name(self) -> str:
        traj_name = self.trajectory_name
        grid_name = self.full_grid.get_name()
        return f"{traj_name}_{grid_name}"

    def read_fg_PT(self) -> FullGrid:
        """
        If no FG is provided as input but the input is a pseutotrajectory, the grid that was used in PT generation
        will also be used in assignments.
        """

        input_names = None
        full_grid_name = None

        # first step: read the name of the full grid from the log file
        try:
            with open(f"{PATH_OUTPUT_LOGGING}{self.trajectory_name}.log") as f:
                while input_names is None or full_grid_name is None:
                    line = f.readline()
                    if line.startswith("INFO:PtLogger:input grid parameters:"):
                        input_names = line.strip().split(": ")[-1]
                    elif line.startswith("INFO:PtLogger:full grid name:"):
                        full_grid_name = line.strip().split(": ")[-1]
        except FileNotFoundError:
            raise ValueError("If you are inputing a trajectory (not PT), you must provide a FullGrid")

        self.grid_name = full_grid_name

        input_names = input_names.split(" ")
        t_input = " ".join(input_names[2:])
        fg = FullGrid(o_grid_name=input_names[0], b_grid_name=input_names[1], t_grid_name=t_input,
                      use_saved=self.use_saved)
        # second step: load the .npy file with the found name
        used_grid = np.load(f"{PATH_OUTPUT_AUTOSAVE}get_full_grid_as_array_{full_grid_name}.npy")

        # third step: assert that this is actually the grid that has been used
        assert np.allclose(used_grid, fg.get_full_grid_as_array())

        return fg

    """
    --------------------------------------------------------------------------------------------------
                               Getters to obtain important indices.
    --------------------------------------------------------------------------------------------------
    """

    def get_indices_k_lowest_energies(self, k: int, energy_type: str):
        all_energies = self.get_magnitude_energy(energy_type)
        return k_argmin_in_array(all_energies, k)

    def get_indices_neighbours_of_cell_i(self, i: int) -> NDArray:
        adj_array = csr_array(self.full_grid.get_full_adjacency())[:, [i]].toarray().T[0]
        neighbour_cell_indices = np.nonzero(adj_array)[0]
        neighbour_indices = []
        for cell_i in neighbour_cell_indices:
            neighbour_indices.extend(self.get_indices_same_cell(cell_i))
        return np.array(neighbour_indices)

    def get_indices_same_orientation(self, quaternion_grid_index: int):
        return np.where(self.get_quaternion_assignments() == quaternion_grid_index)[0]

    def get_indices_same_position(self, position_grid_index: int):
        return np.where(self.get_position_assignments() == position_grid_index)[0]

    def get_indices_same_cell(self, full_grid_index: int):
        return np.where(self.get_full_assignments() == full_grid_index)[0]

    """
    --------------------------------------------------------------------------------------------------
                               Getters to obtain a measure of magnitude.
    --------------------------------------------------------------------------------------------------
    """

    def get_magnitude_energy(self, energy_type: str):
        if self.energies is None:
            self.energies = XVGParser(f"{PATH_OUTPUT_ENERGIES}{self.trajectory_name}.xvg").get_parsed_energy()
        return self.energies.get_energies(energy_type)

    def get_magnitude_ith_eigenvector(self, i: int):
        evalu, evec = self.get_transition_model().get_eigenval_eigenvec()
        my_eigenvector = evec[0].T[i]
        return my_eigenvector

    def get_position_assignments(self):
        if self.position_assignments is None:
            self.position_assignments = self._assign_trajectory_2_position_grid()
        return self.position_assignments

    def get_quaternion_assignments(self):
        if self.quaternion_assignments is None:
            self.quaternion_assignments = self._assign_trajectory_2_quaternion_grid()
        return self.quaternion_assignments

    def get_full_assignments(self):
        if self.full_assignments is None:
            self.full_assignments = self._assign_trajectory_2_full_grid()
        return self.full_assignments

    def get_transition_model(self, tau_array=None, energy_type="Potential"):
        if self.transition_model is None:
            if self.is_pt:
                self.transition_model = SQRA(self, energy_type=energy_type, use_saved=self.use_saved)
            else:
                self.transition_model = MSM(self, tau_array=tau_array, use_saved=self.use_saved)
        return self.transition_model


    def _assign_trajectory_2_quaternion_grid(self):
        """
        Assign every frame of the trajectory to the closest quaternion from the b_grid_points.
        """
        # find PA and direction of reference structure
        reference_principal_axes = self.reference_universe.atoms.principal_axes().T
        inverse_pa = np.linalg.inv(reference_principal_axes)
        reference_direction = determine_positive_directions(self.reference_universe, "all")

        # find PA and direction along trajectory
        pa_frames = AnalysisFromFunction(lambda ag: ag.principal_axes().T, self.trajectory_universe.trajectory,
                                         self.trajectory_universe.select_atoms(self.second_molecule_selection))
        pa_frames.run()
        pa_frames = pa_frames.results['timeseries']

        direction_frames = AnalysisFromFunction(lambda ag: np.tile(reference_direction*determine_positive_directions(ag, "all"), (3, 1)),
                                                self.trajectory_universe.trajectory,
                                                self.trajectory_universe.select_atoms(self.second_molecule_selection))
        direction_frames.run()
        direction_frames = direction_frames.results['timeseries']
        directed_pas = np.multiply(pa_frames, direction_frames)
        produkt = np.matmul(directed_pas, inverse_pa)
        # get the quaternions that caused the rotation from reference to each frame
        calc_quat = np.round(Rotation.from_matrix(produkt).as_quat(), 6)
        # now using a quaternion metric, determine which of b_grid_points is closest

        b_grid_points = self.full_grid.b_rotations.get_grid_as_array()
        b_indices = np.argmin(cdist(b_grid_points, calc_quat, metric=distance_between_quaternions), axis=0)
        # almost everything correct but the order is somehow mixed???
        return b_indices

    def _assign_trajectory_2_t_grid(self) -> NDArray:
        """
        Given a trajectoryand an array of available radial (t-grid) points, assign each frame of the trajectory
        to the closest radial point.

        Returns:
            an integer array as long as the trajectory, each element an index of the closest point of the radial grid
            like [0, 0, 0, 1, 1, 1, 2 ...] (for a PT with 3 orientations)
        """
        t_grid_points = self.full_grid.get_radii()
        t_selection = AnalysisFromFunction(
            lambda ag: np.argmin(np.abs(t_grid_points - np.linalg.norm(ag.center_of_mass())), axis=0),
            self.trajectory_universe.trajectory,
            self.trajectory_universe.select_atoms(self.second_molecule_selection))
        t_selection.run()
        t_indices = t_selection.results['timeseries'].flatten()
        return t_indices

    def _assign_trajectory_2_o_grid(self) -> NDArray:
        """
        Assign every frame of the trajectory (or PT) to the best fitting point of position grid

        Returns:
            an array of position grid indices
        """
        o_grid_points = self.full_grid.position_grid.o_rotations.get_grid_as_array()
        # now using a normalized com and a metric on a sphere, determine which of o_grid_points is closest
        o_selection = AnalysisFromFunction(lambda ag: np.argmin(cdist(o_grid_points, normalise_vectors(
            ag.center_of_mass())[np.newaxis, :], metric="cos"), axis=0),
                                           self.trajectory_universe.trajectory,
                                           self.trajectory_universe.select_atoms(self.second_molecule_selection))
        o_selection.run()
        o_indices = o_selection.results['timeseries'].flatten()
        return o_indices

    def _assign_trajectory_2_position_grid(self):
        """
        Combine assigning to t_grid and o_grid.
        """
        t_assignments = self._assign_trajectory_2_t_grid()
        o_assignments = self._assign_trajectory_2_o_grid()
        # sum up the layer index and o index correctly
        return np.array(t_assignments * self.full_grid.get_o_N() + o_assignments, dtype=int)

    def _assign_trajectory_2_full_grid(self):
        return self.get_position_assignments() * self.full_grid.get_b_N() + self.get_quaternion_assignments()



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
        self.assignments = sim_hist.get_full_assignments()
        self.num_cells = len(self.sim_hist.full_grid)
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
    def get_eigenval_eigenvec(self, num_eigenv: int = 8, **kwargs) -> Tuple[NDArray, NDArray]:
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
        for tau_i, tau in enumerate(tqdm(self.tau_array)):
            tm = all_tms[tau_i]  # the transition matrix for this tau
            #tm[np.isnan(tm)] = 0  # replace nans with zeros
            # in order to compute left eigenvectors, compute right eigenvectors of the transpose
            if isinstance(self, MSM):
                sigma=None # or nothing??
            elif isinstance(self, SQRA):
                sigma=1
            else:
                raise TypeError(f"When not MSM and not SQRA, what then? {type(self)}")
            eigenval, eigenvec = eigs(tm.T, num_eigenv, maxiter=100000, tol=0, which="LM", sigma=sigma, **kwargs) #, sigma=1
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
            self.transition_matrix = np.zeros(shape=(self.num_tau,), dtype=object)
            for tau_i, tau in enumerate(tqdm(self.tau_array)):
                sparse_count_matrix = dok_array((self.num_cells, self.num_cells))
                # save the number of transitions between cell with index i and cell with index j
                #count_per_cell = {(i, j): 0 for i in range(self.num_cells) for j in range(self.num_cells)}
                if not noncorr:
                    window_cell = window(self.assignments, int(tau))
                else:
                    window_cell = noncorr_window(self.assignments, int(tau))
                for cell_slice in window_cell:
                    try:
                        el1, el2 = cell_slice
                        sparse_count_matrix[el1, el2] += 1
                        # enforce detailed balance
                        sparse_count_matrix[el2, el1] += 1
                    except KeyError:
                        # the point is outside the grid and assigned to NaN - ignore for now
                        pass
                sparse_count_matrix = sparse_count_matrix.tocsr()
                sums = sparse_count_matrix.sum(axis=1)
                # to avoid dividing by zero
                sums[sums == 0] = 1
                # now dividing with counts (actually multiplying with inverse)
                diagonal_values = np.reciprocal(sums)
                diagonal_matrix = diags(diagonal_values, format='csr')

                # Left multiply the CSR matrix with the diagonal matrix
                self.transition_matrix[tau_i] = diagonal_matrix.dot(sparse_count_matrix)
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
            obtained_energies = self.sim_hist.get_magnitude_energy(energy_type=self.energy_type)
            for a, e in zip(self.assignments, obtained_energies):
                if not np.isnan(a):
                    all_energies[int(a)] += e
                    energy_counts[int(a)] += 1
            # in both cases avoiding division with zero
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
    #H2O_full_grid = FullGrid(b_grid_name="25", o_grid_name="1", t_grid_name="0.4", use_saved=True)
    H2O_sh = SimulationHistogram("H2O_H2O_0493", "H2O", is_pt=True, full_grid=None,
                                 second_molecule_selection="bynum 4:6")
    quat_assignents = H2O_sh.get_quaternion_assignments()
    print(quat_assignents)
    print(H2O_sh.full_grid.get_quaternion_index())




