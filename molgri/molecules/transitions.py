"""
Combine a ParsedTrajectory and a FullGrid to generate a MSM or SqRA model.

In this module, the two methods of evaluating transitions between states - the MSM and the SqRA approach - are
implemented.
"""
from typing import Optional, Tuple, Sequence, Any
from multiprocessing import Pool
from functools import partial
from time import time

import numpy as np
import pandas as pd
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis import Universe
from numpy.typing import NDArray
from scipy.sparse import coo_array, csr_array, diags, dok_array
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.constants import k as kB, N_A
import MDAnalysis.transformations as trans


from molgri.molecules.rate_merger import delete_rate_cells, determine_rate_cells_to_join, \
    determine_rate_cells_with_too_high_energy, \
    merge_matrix_cells
from molgri.space.fullgrid import from_full_array_to_o_b_t


from molgri.space.utils import normalise_vectors


class AssignmentTool:
    """
    This tool is used to assign trajectory frames to grid cells.
    """

    def __init__(self, full_array: NDArray, full_universe: Universe, second_molecule: Universe,
                 stop=None, include_outliers = False, cartesian_grid = True):
        self.full_array = full_array
        # these grids are also in A
        self.o_array, self.b_array, self.t_array = from_full_array_to_o_b_t(self.full_array)
        # whatever the format, MDAnalysis automatically converts to A
        self.trajectory_universe = full_universe
        self.reference_universe = second_molecule
        self.second_molecule_selection = self._determine_second_molecule()
        first_molecule = self.trajectory_universe.select_atoms("not " + self.second_molecule_selection)
        self.trajectory_universe.trajectory.add_transformations(trans.translate(-first_molecule.center_of_mass()))
        self.include_outliers = include_outliers
        self.cartesian_grid = cartesian_grid
        self.stop = stop
        if stop is None:
            self.stop = len(self.trajectory_universe.trajectory)

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

    def _complex_mdanalysis_func(self, frame_index, ag, reference_direction):
        # index the trajectory to set it to the frame_index frame
        ag.universe.trajectory[frame_index]
        return np.multiply(ag.principal_axes().T, np.tile(self._determine_positive_directions(ag) / reference_direction, (3, 1)))

    def _get_rotation_matrices(self):
        """
        For every frame in the trajectory, find the exact quaternion needed to go from reference structure of
        molecule2 to structure of molecule2 in this frame (this is exact if trajectory is done with rigid molecules,
        otherwise it must be an approximation).
        """
        reference_principal_axes = self.reference_universe.atoms.principal_axes().T
        inverse_pa = np.linalg.inv(reference_principal_axes)
        reference_direction = self._determine_positive_directions(self.reference_universe)
        if reference_direction is None:
            raise ValueError("All atoms perpendicular to at least one of principal axes, can't determine direction.")
        """
        I am very sorry that this part is optimized, parallelized and completely unreadable. Rely on tests to make
        sure this stuff works and be happy that your calculations are no longer taking two days.
        """
        run_per_frame = partial(self._complex_mdanalysis_func,
                                ag=self.trajectory_universe.select_atoms(self.second_molecule_selection),
                                reference_direction=reference_direction)
        frame_values = np.arange(stop=self.stop)
        with Pool(1) as worker_pool:
            direction_frames = worker_pool.map(run_per_frame, frame_values)
        # t_selection = AnalysisFromFunction(
        #     lambda ag: self._complex_mdanalysis_func,
        #     self.trajectory_universe.trajectory,
        #     self.trajectory_universe.select_atoms(self.second_molecule_selection))
        # t_selection.run(stop=self.stop)  # frames=self.within_R
        # direction_frames =t_selection.results['timeseries']
        # print(direction_frames[0].shape)
        return np.matmul(direction_frames, inverse_pa)

    def _get_quaternion_assignments(self):
        """
        Assign every perfect-fit quaternion to the closest-available quaternion from the b_grid_points.
        """
        t1=time()
        my_quaternions = self._get_rotation_matrices()
        """
        Explanation: we have N_traj_len produkt matrices called P_i and N_quat reference matrices called R_i. The
        matrix product R_i@P_i.T describes the rotation matrix needed to get from R_i to P_i. We want the magnitude
        of this rotation to be as small as possible.

        So we calculate the matrix of magnitudes of size N_quat x N_traj_len and select the index of the smallest
        magnitude per row.

        This part of the function should be pretty fast.
        """
        reference_matrices = Rotation(self.b_array).as_matrix()
        alignment_magnitudes = np.empty((len(reference_matrices), len(my_quaternions)))
        for i, rm in enumerate(reference_matrices):
            alignment_magnitudes[i] = Rotation.from_matrix(rm@my_quaternions.transpose(0, 2, 1)).magnitude()
        result = np.argmin(alignment_magnitudes, axis=0).flatten()
        t2= time()
        print("QUAT STATISTICS", pd.DataFrame(result).describe())
        return np.array(result)

    def _t_assignment_function(self, ag):
        min_dist_to_gridpoint = np.argmin(np.abs(self.t_array - np.linalg.norm(ag.center_of_mass())))
        if self.include_outliers:
            return min_dist_to_gridpoint
        else:
            outer_bound = self.t_array[-1] + 0.5 * (self.t_array[-1] - self.t_array[-2])
            if np.linalg.norm(ag.center_of_mass()) > outer_bound:
                return np.nan
            else:
                return min_dist_to_gridpoint

    def _get_t_assignments(self) -> NDArray:
        """
        Given a trajectoryand an array of available radial (t-grid) points, assign each frame of the trajectory
        to the closest radial point.
        Returns:
            an integer array as long as the trajectory, each element an index of the closest point of the radial grid
            like [0, 0, 0, 1, 1, 1, 2 ...] (for a PT with 3 orientations)
        """
        t_selection = AnalysisFromFunction(
            lambda ag: self._t_assignment_function(ag),
            self.trajectory_universe.trajectory,
            self.trajectory_universe.select_atoms(self.second_molecule_selection))
        t_selection.run(stop=self.stop)
        t_indices = t_selection.results['timeseries'].flatten()
        print("T STATISTICS", pd.DataFrame(t_indices).describe())
        return t_indices

    def _o_assignment_function(self, ag):
        if self.cartesian_grid:
            metric = "euclidean"
        else:
            metric = "cos"

        all_possible_distances = cdist(self.o_array, normalise_vectors(
            ag.center_of_mass()[np.newaxis, :]), metric=metric)
        result = np.argmin(all_possible_distances, axis=0)
        return result


    def _get_o_assignments(self) -> NDArray:
        """
        Assign every frame of the trajectory (or PT) to the best fitting point of position grid
        Returns:
            an array of position grid indices
        """
        # now using a normalized com and a metric on a sphere, determine which of o_grid_points is closest
        o_selection = AnalysisFromFunction(lambda ag: self._o_assignment_function(ag),
                                           self.trajectory_universe.trajectory,
                                           self.trajectory_universe.select_atoms(self.second_molecule_selection))
        o_selection.run(stop=self.stop)
        o_indices = o_selection.results['timeseries'].flatten()
        print("O STATISTICS", pd.DataFrame(o_indices).describe())
        return o_indices

    def _get_position_assignments(self):
        """
        Combine assigning to t_grid and o_grid.
        """
        t_assignments = self._get_t_assignments()
        o_assignments = self._get_o_assignments()
        # sum up the layer index and o index correctly
        return np.array(t_assignments * len(self.o_array) + o_assignments, dtype=float)

    def get_full_assignments(self) -> NDArray:
        return self._get_position_assignments() * len(self.b_array) + self._get_quaternion_assignments()


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
            yield tuple([int(el) for el in start_stop_list if not np.isnan(el)])


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

    def get_one_tau_transition_matrix(self, tau: float, noncorrelated_windows: bool):
        sparse_count_matrix = dok_array((self.total_num_cells, self.total_num_cells))
        # save the number of transitions between cell with index i and cell with index j
        # count_per_cell = {(i, j): 0 for i in range(self.num_cells) for j in range(self.num_cells)}
        if noncorrelated_windows:
            window_cell = noncorr_window(self.assigned_trajectory, int(tau))
        else:
            window_cell = window(self.assigned_trajectory, int(tau))

        for cell_slice in window_cell:
            if len(cell_slice)<2:
                pass
            else:
                el1, el2 = cell_slice
                sparse_count_matrix[el1, el2] += 1
                # enforce detailed balance
                sparse_count_matrix[el2, el1] += 1
            # except IndexError:
            #     raise IndexError(f"Assignments outside of FullGrid borders: {cell_slice} > "
            #                      f"{self.total_num_cells, self.total_num_cells}")
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
            transition_matrix[tau_i] = self.get_one_tau_transition_matrix(tau, noncorrelated_windows=noncorrelated_windows)
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
        diff_energies = self.energies[transition_matrix.row] - self.energies[transition_matrix.col]
        # cannot allow more than 3 orders of magnitude difference
        print(f"Warning! {len(np.where(diff_energies > 5e2)[0])} pairs of cells have a very large difference in "
              f"energy, more than factor 500. This would lead to overflow, so these differences are capped to a "
              f"factor 500. This might be a sign of poor discretisation or just the case of L-J overlap.")
        diff_energies = np.where(diff_energies < 5e2, diff_energies, 5e2)

        pi_exponent = np.round(diff_energies,14) * 1000 / (2 * kB * N_A * T)


        transition_matrix.data *= np.exp(pi_exponent)
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

    def __init__(self, matrix_to_decompose):
        """

        Args:
            matrix_to_decompose (): either a single matrix or an array of matrices (for different taus) we want to
            decompose
        """
        self.matrix_to_decompose = matrix_to_decompose

    def get_decomposition(self, tol: float, maxiter: int, which: str, sigma: Optional[float], k=12):
        """
        The function for users - will decompose all matrices.

            tol ():
            maxiter ():
            which ():
            sigma ():

        Returns:

        """
        eigenval, eigenvec = eigs(self.matrix_to_decompose.T, k=k, tol=tol, maxiter=maxiter, which=which, sigma=sigma)
        # if imaginary eigenvectors or eigenvalues, raise error
        if not np.allclose(eigenvec.imag.max(), 0, rtol=1e-3, atol=1e-5) or not np.allclose(eigenval.imag.max(), 0,
                                                                                            rtol=1e-3, atol=1e-5):
            print(f"Complex values for eigenvectors and/or eigenvalues: {eigenvec}, {eigenval}")
        eigenvec = eigenvec.real
        eigenval = eigenval.real
        # sort eigenvectors according to their eigenvalues
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]
        return eigenval, eigenvec
