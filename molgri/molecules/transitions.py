"""
Combine a ParsedTrajectory and a FullGrid to generate a MSM or SqRA model.

In this module, the two methods of evaluating transitions between states - the MSM and the SqRA approach - are
implemented.
"""
from typing import Optional, Tuple, Sequence, Any

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from numpy.typing import NDArray
from scipy.sparse import coo_array, csr_array, diags, dok_array
from scipy.sparse.linalg import eigs, eigsh
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.constants import k as kB, N_A

from molgri.molecules.rate_merger import delete_rate_cells, determine_rate_cells_to_join, \
    determine_rate_cells_with_too_high_energy, \
    merge_matrix_cells
from molgri.space.fullgrid import from_full_array_to_o_b_t


from molgri.space.utils import distance_between_quaternions, normalise_vectors


class AssignmentTool:
    """
    This tool is used to assign trajectory frames to grid cells.
    """

    def __init__(self, full_array: NDArray, path_structure: str, path_trajectory: str, path_reference_m2: str):
        self.full_array = full_array
        # these grids are also in A
        self.o_array, self.b_array, self.t_array = from_full_array_to_o_b_t(self.full_array)
        # whatever the format, MDAnalysis automatically converts to A
        self.trajectory_universe = mda.Universe(path_structure, path_trajectory)
        self.reference_universe = mda.Universe(path_reference_m2)
        self.second_molecule_selection = self._determine_second_molecule()

    def _determine_second_molecule(self):
        num_atoms_total = len(self.trajectory_universe.atoms)
        num_atoms_m2 = len(self.reference_universe.atoms)
        # indexing in MDAnalysis is 1-based
        # we look for indices of the second molecule
        num_atoms_m1 = num_atoms_total - num_atoms_m2
        # indices are inclusive
        return f"bynum  {num_atoms_m1+1}:{num_atoms_total+1}"

    def _determine_positive_directions(self, current_universe):
        pas = current_universe.atoms.principal_axes()
        com = current_universe.atoms.center_of_mass()
        directions = [0, 0, 0]
        for atom_pos in current_universe.atoms.positions:
            for i, pa in enumerate(pas):
                # need to round to avoid problems - assigning direction with atoms very close to 0
                cosalpha = np.round(pa.dot(atom_pos - com), 6)
                directions[i] = np.sign(cosalpha)
            if not np.any(np.isclose(directions, 0)):
                break
        # if exactly one unknown use the other two and properties of righthanded systems to get third
        if np.sum(np.isclose(directions, 0)) == 1:
            # only these combinations of directions are possible in righthanded coordinate systems
            allowed_righthanded = [[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]]
            for ar in allowed_righthanded:
                # exactly two identical (and the third is zero)
                if np.sum(np.isclose(ar, directions)) == 2:
                    directions = ar
                    break
        # if two (or three - that would just be an atom) unknowns raise an error
        elif np.sum(np.isclose(directions, 0)) > 1:
            raise ValueError("All atoms perpendicular to at least one of principal axes, can´t determine direction.")
        return np.array(directions)

    def _get_quaternion_assignments(self):
        """
        Assign every frame of the trajectory to the closest quaternion from the b_grid_points.
        """
        # find PA and direction of reference structure
        reference_principal_axes = self.reference_universe.atoms.principal_axes().T
        inverse_pa = np.linalg.inv(reference_principal_axes)
        reference_direction = self._determine_positive_directions(self.reference_universe)

        # find PA and direction along trajectory
        pa_frames = AnalysisFromFunction(lambda ag: ag.principal_axes().T, self.trajectory_universe.trajectory,
                                         self.trajectory_universe.select_atoms(self.second_molecule_selection))
        pa_frames.run()
        pa_frames = pa_frames.results['timeseries']

        direction_frames = AnalysisFromFunction(lambda ag: np.tile(self._determine_positive_directions(
            ag) / reference_direction, (3, 1)),
                                                self.trajectory_universe.trajectory,
                                                self.trajectory_universe.select_atoms(self.second_molecule_selection))
        direction_frames.run()
        direction_frames = direction_frames.results['timeseries']
        directed_pas = np.multiply(pa_frames, direction_frames)
        produkt = np.matmul(directed_pas, inverse_pa)
        # get the quaternions that caused the rotation from reference to each frame
        calc_quat = np.round(Rotation.from_matrix(produkt).as_quat(), 6)
        b_indices = np.argmin(cdist(self.b_array, calc_quat, metric=distance_between_quaternions), axis=0)
        # almost everything correct but the order is somehow mixed???
        return b_indices

    def _get_t_assignments(self) -> NDArray:
        """
        Given a trajectoryand an array of available radial (t-grid) points, assign each frame of the trajectory
        to the closest radial point.

        Returns:
            an integer array as long as the trajectory, each element an index of the closest point of the radial grid
            like [0, 0, 0, 1, 1, 1, 2 ...] (for a PT with 3 orientations)
        """
        t_selection = AnalysisFromFunction(
            lambda ag: np.argmin(np.abs(self.t_array - np.linalg.norm(ag.center_of_mass())), axis=0),
            self.trajectory_universe.trajectory,
            self.trajectory_universe.select_atoms(self.second_molecule_selection))
        t_selection.run()
        t_indices = t_selection.results['timeseries'].flatten()
        return t_indices

    def _get_o_assignments(self) -> NDArray:
        """
        Assign every frame of the trajectory (or PT) to the best fitting point of position grid

        Returns:
            an array of position grid indices
        """
        # now using a normalized com and a metric on a sphere, determine which of o_grid_points is closest
        o_selection = AnalysisFromFunction(lambda ag: np.argmin(cdist(self.o_array, normalise_vectors(
            ag.center_of_mass())[np.newaxis, :], metric="cos"), axis=0),
                                           self.trajectory_universe.trajectory,
                                           self.trajectory_universe.select_atoms(self.second_molecule_selection))
        o_selection.run()
        o_indices = o_selection.results['timeseries'].flatten()
        return o_indices

    def _get_position_assignments(self):
        """
        Combine assigning to t_grid and o_grid.
        """
        t_assignments = self._get_t_assignments()
        o_assignments = self._get_o_assignments()
        # sum up the layer index and o index correctly
        return np.array(t_assignments * len(self.o_array) + o_assignments, dtype=int)

    def get_full_assignments(self) -> NDArray:
        return self._get_position_assignments() * len(self.b_array) + self._get_quaternion_assignments()


# class SimulationHistogram:
#
#     """
#     This class takes the data from ParsedTrajectory and combines it with a specific FullGrid so that each simulation
#     step can be assigned to a cell and the occupancy of those cells evaluated.
#     """
#
#     def __init__(self, trajectory_name: str, reference_name: str, is_pt: bool, second_molecule_selection: str, full_grid: FullGrid = None,
#                  use_saved=True):
#         """
#         Situation: you have created a (pseudo)trajectory (.gro and .xtc files in PATH_OUTPUT_PT) and calculated the
#         energies for every frame (.xvg file PATH_OUTPUT_ENERGIES) - all files have the name given by trajectory_name.
#         Now you want to assign one of the molecules (selected by second_molecule_selection) in every frame to one
#         cell ub full_grid.
#
#         Args:
#             trajectory_name (str): like H2O_H2O_0095 for PTs or H2O_H2O_0095_1000 for trajectories, the folder
#                                    PATH_OUTPUT_PT will be searched for this name
#             full_grid (FullGrid): used to assign to cells; even if the input is a PT, you can use a different FullGrid
#                                   but if None, the FullGrid used in creation will be used
#             second_molecule_selection (str): will be forwarded to trajectory_universe to identify the moving molecule
#
#         """
#         self.trajectory_name = trajectory_name
#         split_name = self.trajectory_name.strip().split("_")
#         self.m1_name = split_name[0]
#         self.m2_name = split_name[1]
#         self.is_pt = is_pt
#         if self.is_pt:
#             self.structure_name = self.trajectory_name
#         else:
#             self.structure_name = "_".join(self.trajectory_name.split("_")[:-1])
#         self.use_saved = use_saved
#         self.second_molecule_selection = second_molecule_selection
#         self.energies = None
#         try:
#             self.trajectory_universe = mda.Universe(f"{PATH_OUTPUT_PT}{self.structure_name}.gro",
#                                                     f"{PATH_OUTPUT_PT}{self.trajectory_name}.trr")
#         except:
#             self.trajectory_universe = mda.Universe(f"{PATH_OUTPUT_PT}{self.structure_name}.gro",
#                                                     f"{PATH_OUTPUT_PT}{self.trajectory_name}.xtc")
#         try:
#             self.reference_universe = mda.Universe(f"{PATH_INPUT_BASEGRO}{reference_name}.gro")
#         except:
#             self.reference_universe = mda.Universe(f"{PATH_INPUT_BASEGRO}{reference_name}.xyz")
#         if full_grid is None and self.is_pt:
#             full_grid = self.read_fg_PT()
#         self.full_grid = full_grid
#
#         # assignments
#         self.position_assignments = None
#         self.quaternion_assignments = None
#         self.full_assignments = None
#         self.transition_model = None
#
#     def __len__(self):
#         return len(self.trajectory_universe.trajectory)
#
#
#     def get_indices_k_lowest_energies(self, k: int, energy_type: str):
#         all_energies = self.get_magnitude_energy(energy_type)
#         return k_argmin_in_array(all_energies, k)
#
#     def get_indices_neighbours_of_cell_i(self, i: int) -> NDArray:
#         adj_array = csr_array(self.full_grid.get_full_adjacency())[:, [i]].toarray().T[0]
#         neighbour_cell_indices = np.nonzero(adj_array)[0]
#         neighbour_indices = []
#         for cell_i in neighbour_cell_indices:
#             neighbour_indices.extend(self.get_indices_same_cell(cell_i))
#         return np.array(neighbour_indices)
#
#     def get_indices_same_orientation(self, quaternion_grid_index: int):
#         return np.where(self.get_quaternion_assignments() == quaternion_grid_index)[0]
#
#     def get_indices_same_position(self, position_grid_index: int):
#         return np.where(self.get_position_assignments() == position_grid_index)[0]
#
#     def get_indices_same_cell(self, full_grid_index: int):
#         return np.where(self.get_full_assignments() == full_grid_index)[0]
#
#     """
#     --------------------------------------------------------------------------------------------------
#                                Getters to obtain a measure of magnitude.
#     --------------------------------------------------------------------------------------------------
#     """
#
#     def get_magnitude_energy(self, energy_type: str):
#         if self.energies is None:
#             self.energies = XVGParser(f"{PATH_OUTPUT_ENERGIES}{self.trajectory_name}.xvg").get_parsed_energy()
#         return self.energies.get_energies(energy_type)
#
#     def get_magnitude_ith_eigenvector(self, i: int):
#         evalu, evec = self.get_transition_model().get_eigenval_eigenvec()
#         my_eigenvector = evec[0].T[i]
#         return my_eigenvector


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


class MSM:

    """
    From assignments create a MSM transition matrix
    """

    def __init__(self, assigned_trajectory: NDArray, total_num_cells: int):
        self.assigned_trajectory = assigned_trajectory
        self.total_num_cells = total_num_cells

    def _get_one_tau_transition_matrix(self, tau: float, noncorrelated_windows: bool):
        sparse_count_matrix = dok_array((self.total_num_cells, self.total_num_cells))
        # save the number of transitions between cell with index i and cell with index j
        # count_per_cell = {(i, j): 0 for i in range(self.num_cells) for j in range(self.num_cells)}
        if noncorrelated_windows:
            window_cell = noncorr_window(self.assigned_trajectory, int(tau))
        else:
            window_cell = window(self.assigned_trajectory, int(tau))

        for cell_slice in window_cell:
            try:
                el1, el2 = cell_slice
                sparse_count_matrix[el1, el2] += 1
                # enforce detailed balance
                sparse_count_matrix[el2, el1] += 1
            except IndexError:
                raise IndexError(f"Assignments outside of FullGrid borders: {cell_slice}")
        sparse_count_matrix = sparse_count_matrix.tocsr()
        sums = sparse_count_matrix.sum(axis=1)
        # to avoid dividing by zero
        sums[sums == 0] = 1
        # now dividing with counts (actually multiplying with inverse)
        diagonal_values = np.reciprocal(sums)
        diagonal_matrix = diags(diagonal_values, format='csr')
        # Left multiply the CSR matrix with the diagonal matrix
        return diagonal_matrix.dot(sparse_count_matrix)

    def get_all_tau_transition_matrices(self, taus: NDArray, noncorrelated_windows: bool):
        transition_matrix = np.zeros(shape=taus.shape, dtype=object)
        for tau_i, tau in enumerate(taus):
            transition_matrix[tau_i] = self._get_one_tau_transition_matrix(tau, noncorrelated_windows=noncorrelated_windows)
        return transition_matrix


class SQRA:

    def __init__(self, energies: NDArray, volumes: NDArray, distances: csr_array, surfaces: csr_array):
        self.energies = energies
        self.volumes = volumes
        self.distances = distances
        self.surfaces = surfaces

    def get_rate_matrix(self, D: float, T: float) -> csr_array:
        # calculating rate matrix
        # for sqra demand that each energy corresponds to exactly one cell
        assert len(self.energies) == len(self.volumes), f"{len(self.energies)} != {len(self.volumes)}"
        # you cannot multiply or divide directly in a coo format
        transition_matrix = D * self.surfaces  #/ all_distances
        transition_matrix = transition_matrix.tocoo()
        transition_matrix.data /= self.distances.tocoo().data
        # Divide every row of transition_matrix with the corresponding volume
        transition_matrix.data /= self.volumes[transition_matrix.row]
        # multiply with sqrt(pi_j/pi_i) = e**((V_i-V_j)*1000/(2*k_B*N_A*T))
        # gromacs uses kJ/mol as energy unit, boltzmann constant is J/K
        transition_matrix.data *= np.exp(np.round((self.energies[
                                                            transition_matrix.row] - self.energies[
                                                            transition_matrix.col]),14) * 1000 / (
                                                          2 * kB * N_A * T))
        # normalise rows
        sums = transition_matrix.sum(axis=1)
        sums = np.array(sums).squeeze()
        all_i = np.arange(len(self.volumes))
        diagonal_array = coo_array((-sums, (all_i, all_i)), shape=(len(all_i), len(all_i)))
        transition_matrix = transition_matrix.tocsr() + diagonal_array.tocsr()
        return transition_matrix

    def cut_and_merge(self, transition_matrix: csr_array, T: float, lower_limit: float, upper_limit: float) -> tuple:
        # cut and merge
        if lower_limit is not None:
            rate_to_join = determine_rate_cells_to_join(self.distances, self.energies,
                bottom_treshold=lower_limit, T=T)
            transition_matrix, current_index_list = merge_matrix_cells(my_matrix=transition_matrix,
                all_to_join=rate_to_join, index_list=None)
        else:
            current_index_list = None
        if upper_limit is not None:
            too_high = determine_rate_cells_with_too_high_energy(self.energies,energy_limit=upper_limit,T=T)
            transition_matrix, current_index_list = delete_rate_cells(transition_matrix, to_remove=too_high,
                index_list=current_index_list)
        else:
            current_index_list = None
        return transition_matrix, current_index_list


class DecompositionTool:

    def __init__(self, matrix_to_decompose: NDArray, is_msm: bool):
        """

        Args:
            matrix_to_decompose (): either a single matrix or an array of matrices (for different taus) we want to
            decompose
        """
        self.matrix_to_decompose = matrix_to_decompose
        self.is_msm = is_msm  # in this case multiple matrices for multiple taus

    def get_decomposition(self, tol: float, maxiter: int, which: str, sigma: Optional[float]):
        """
        The function for users - will decompose all matrices if self.matrix_to_decompose is 3D, else the one matrix
        Args:
            tol ():
            maxiter ():
            which ():
            sigma ():

        Returns:

        """
        if self.is_msm:
            all_eigenval = []
            all_eigenvec = []
            for submatrix in self.matrix_to_decompose:
                one_eigenval, one_eigenvec = self._single_decomposition(submatrix, tol=tol, maxiter=maxiter,
                                                                       which=which, sigma=sigma)
                all_eigenval.append(one_eigenval)
                all_eigenvec.append(one_eigenvec)
            return all_eigenval, all_eigenvec
        else:
            return self._single_decomposition(self.matrix_to_decompose, tol=tol, maxiter=maxiter, which=which,
                                              sigma=sigma)

    def _single_decomposition(self, my_matrix, **kwargs):
        eigenval, eigenvec = eigs(my_matrix.T, **kwargs)
        # if imaginary eigenvectors or eigenvalues, raise error
        if not np.allclose(eigenvec.imag.max(), 0, rtol=1e-3, atol=1e-5) or not np.allclose(eigenval.imag.max(), 0,
                                                                                            rtol=1e-3, atol=1e-5):
            raise ValueError(f"Complex vlues for eigenvectors and/or eigenvalues: {eigenvec}, {eigenval}")
        eigenvec = eigenvec.real
        eigenval = eigenval.real
        # sort eigenvectors according to their eigenvalues
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]
        return eigenval, eigenvec


