"""
Combine a ParsedTrajectory and a FullGrid to generate a MSM or SqRA model.

In this module, the two methods of evaluating transitions between states - the MSM and the SqRA approach - are
implemented.
"""
from abc import ABC, abstractmethod
import copy
from typing import Tuple, Sequence, Any

import numpy as np
import MDAnalysis as mda
from IPython.core.display import display
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.align import rotation_matrix
import nglview as nv
from molgri.space.rotations import two_vectors2rot

from numpy.typing import NDArray
from scipy.sparse.linalg import eigs
from scipy.spatial.transform import Rotation
from scipy.constants import k as kB, N_A
from scipy.spatial.distance import cdist
from tqdm import tqdm

from molgri.space.translations import get_between_radii
from molgri.wrappers import save_or_use_saved
from molgri.molecules.parsers import FileParser, ParsedEnergy, ParsedMolecule, ParsedTrajectory
from molgri.space.fullgrid import FullGrid
from molgri.paths import OUTPUT_PLOTTING_DATA, PATH_INPUT_BASEGRO, PATH_OUTPUT_LOGGING, PATH_OUTPUT_PT
from molgri.space.utils import angle_between_vectors, distance_between_quaternions, hemisphere_quaternion_set, \
    norm_per_axis


class SimulationHistogram:

    """
    This class takes the data from ParsedTrajectory and combines it with a specific FullGrid so that each simulation
    step can be assigned to a cell and the occupancy of those cells evaluated.
    """

    def __init__(self, trajectory_path: str, trajectory_name: str, full_grid: FullGrid, energies: ParsedEnergy,
                 second_molecule_selection):
        self.trajectory_name = trajectory_name
        self.second_molecule_selection = second_molecule_selection
        split_name = trajectory_name.split("_")
        self.m1_name = split_name[0]
        self.m2_name = split_name[1]
        self.trajectory_universe = mda.Universe(f"{trajectory_path}{trajectory_name}.gro",
                                                f"{trajectory_path}{trajectory_name}.xtc")
        self.full_grid = full_grid
        self.energies = energies

        # assignments
        self.position_assignments = self._assign_trajectory_2_position_grid()
        self.quaternion_assignments = self._assign_trajectory_2_quaternion_grid()
        self.full_assignments = self._assign_trajectory_2_full_grid()

    # noinspection PyMissingOrEmptyDocstring
    def get_name(self) -> str:
        traj_name = self.trajectory_name
        grid_name = self.full_grid.get_name()
        return f"{traj_name}_{grid_name}"

    def get_position_assignments(self):
        return self.position_assignments

    def get_quaternion_assignments(self):
        return self.quaternion_assignments

    def get_full_assignments(self):
        return self.full_assignments

    # def assign_trajectory_2_quaternion_grid_old(self):
    #
    #     def _extract_universe_second_molecule(original_universe, selection_criteria):
    #         m2 = original_universe.select_atoms(selection_criteria)
    #
    #         coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), m2).run().results['timeseries']
    #         u2 = mda.Merge(m2)
    #         u2.load_new(coordinates, format=MemoryReader)
    #         return u2
    #
    #     # create PT on quaternion-only grid
    #     manager = PtIOManager(self.m1_name, self.m2_name, o_grid_name="1", b_grid_name=self.full_grid.b_grid_name,
    #                           t_grid_name="[0.1]")
    #     manager.construct_pt()
    #
    #     # read the PT universe constructed in previous lines
    #     my_pt_name = manager.get_name()
    #     pt_sec_mol_universe = mda.Universe(f"{PATH_OUTPUT_PT}{my_pt_name}.gro", f"{PATH_OUTPUT_PT}{my_pt_name}.xtc")
    #
    #     # in the real and pt trajectory, extract the second molecule and center it without rotating
    #     trajectory_universe_m2 = _extract_universe_second_molecule(self.trajectory_universe,
    #                                                                self.second_molecule_selection)
    #     pt_universe_m2 = _extract_universe_second_molecule(pt_sec_mol_universe, self.second_molecule_selection)
    #     # move them to center - curently doing that later
    #     # workflow = [mda.transformations.center_in_box(real_traj_sec_mol.atoms, center="mass", point=(0, 0, 0))]
    #     # real_traj_sec_mol.trajectory.add_transformations(*workflow)
    #
    #     # calculate RMSD between each frame of real trajectory and all reference orientations from PT
    #     total_results = []
    #     for i, ts in enumerate(pt_universe_m2.trajectory):
    #         results = []
    #         for j, ts2 in enumerate(trajectory_universe_m2.trajectory):
    #             results.append(rmsd(trajectory_universe_m2.trajectory[j].positions,
    #                                                  pt_universe_m2.trajectory[i].positions,
    #                                                  center=True, weights=trajectory_universe_m2.atoms.masses))
    #         total_results.append(results)
    #
    #     # Of all reference orientations, select the one with the smallest RMSD
    #     total_results = np.array(total_results)
    #     clases = np.argmin(total_results, axis=0)
    #     return clases


    def _assign_trajectory_2_quaternion_grid(self):

        def _extract_universe_second_molecule(original_universe, selection_criteria) -> mda.Universe:
            """
            This function removes the first molecule and rotates the second one to the position [0, 0, 1]
            """
            m2 = original_universe.select_atoms(selection_criteria)

            coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), m2).run().results['timeseries']
            u2 = mda.Merge(m2)
            u2.load_new(coordinates, format=MemoryReader)
            return u2

        def _get_reference_molecule() -> mda.Universe:
            """This function loads the input second molecule and translates it to position [0, 0, 1]"""
            my_file = FileParser(f"{PATH_INPUT_BASEGRO}{self.m2_name}").as_parsed_molecule()
            my_file.translate_to_origin()
            #my_file.translate(np.array([0, 0, 1]))
            # turn to Universe
            u_ref = mda.Merge(my_file.atoms)
            return u_ref

        all_quaternions = self.full_grid.b_rotations.get_grid_as_array()
        print(all_quaternions[:5])



        # in the real and pt trajectory, extract the second molecule and center it without rotating
        trajectory_universe_m2 = _extract_universe_second_molecule(self.trajectory_universe,
                                                                   self.second_molecule_selection)
        # get and center reference structure
        initial_u = _get_reference_molecule()

        # calculate RMSD between each frame of real trajectory and all reference orientations from PT
        total_results = []
        #for i, ts in enumerate(pt_universe_m2.trajectory):
        #    results = []
        from molgri.plotting.widgets import ViewManager

        for j, ts2 in enumerate(trajectory_universe_m2.trajectory):
            trajectory_universe_m2.trajectory[j]


            # first assign closest position grid point so you only consider additional rotation from that point
            my_pos_assignment = int(self.position_assignments[j])
            position = self.full_grid.get_position_grid_as_array()[my_pos_assignment]
            z_vector = np.array([0, 0, np.linalg.norm(position)])

            # something here not right
            fresh_parsed = ParsedMolecule(trajectory_universe_m2.copy().atoms)
            current_positions = trajectory_universe_m2.atoms.positions

            all_rmsd = []
            for i, available_quat in enumerate(all_quaternions):
                current_parsed = ParsedMolecule(trajectory_universe_m2.atoms)
                current_parsed.atoms.positions = current_positions
                print(np.round(current_parsed.get_positions()[0], 3), i)
                rotation_body = Rotation.from_quat(available_quat)
                # TODO: here you could simpy try which of the available quaternion rotations brings you closest to inital
                #  state
                current_parsed.rotate_about_body(rotation_body, inverse=True)
                rotation_origin = Rotation.from_matrix(two_vectors2rot(z_vector, position))
                current_parsed.rotate_about_origin(rotation_origin, inverse=True)
                current_parsed.translate_to_origin()
                all_rmsd.append(rmsd(current_parsed.get_positions(), initial_u.atoms.positions)) #
                # weights=initial_u.atoms.masses)
            best_q = np.argmin(all_rmsd)
            #print(all_rmsd)
            #print(best_q)
            total_results.append(best_q)
            #current_parsed.rotate_to(np.array([0, 0, 1]))
            #print("After o-rotation state")
            #if j%8 == 0:
            #    display(ViewManager(mda.Merge(current_parsed.atoms)).plot_ith_frame(0))

            #rot_mat = rotation_matrix(initial_u.atoms.positions, current_parsed.get_positions(),
            #                          weights=trajectory_universe_m2.atoms.masses)

            #total_results.append(Rotation.from_matrix(rot_mat[0]).as_quat())
        #total_results = np.array(total_results)
        #print(total_results[0])
        #total_results = hemisphere_quaternion_set(total_results)

        #total_results = cdist(total_results, all_quaternions,
        #            metric=distance_between_quaternions)

        #print(total_results[0])
        # Of all reference orientations, select the one with the smallest RMSD

        #clases = np.argmin(total_results.T, axis=0)
        #return clases
        return np.array(total_results)

    def _assign_trajectory_2_position_grid(self) -> NDArray:

        coms = AnalysisFromFunction(lambda ag: ag.center_of_mass(),
                                   self.trajectory_universe.trajectory,
                                    self.trajectory_universe.select_atoms(self.second_molecule_selection))
        coms.run()

        points_vector = coms.results['timeseries']

        rot_points = self.full_grid.o_rotations.get_grid_as_array()
        # this automatically select the one of angles that is < pi
        angles = angle_between_vectors(points_vector, rot_points)
        indices_within_layer = np.argmin(angles, axis=1)

        # determine radii of cells
        norms = norm_per_axis(points_vector)
        layers = np.zeros((len(points_vector),))
        vor_radii = get_between_radii(self.full_grid.t_grid.get_trans_grid())

        # find the index of the layer to which each point belongs
        for i, norm in enumerate(norms):
            for j, vor_rad in enumerate(vor_radii):
                # because norm keeps the shape of the original array
                if norm[0] < vor_rad:
                    layers[i] = j
                    break
            else:
                layers[i] = np.NaN

        layer_len = len(rot_points)
        indices = layers * layer_len + indices_within_layer

        # determine the closest orientation

        return np.array(indices, dtype=int)

    def _assign_trajectory_2_full_grid(self):
        num_quaternions = self.full_grid.b_rotations.get_N()
        return self.get_position_assignments() * num_quaternions + self.get_quaternion_assignments()



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
        self.assignments = self.sim_hist.get_all_assignments()
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
    import sys
    my_fg = FullGrid("8", "12", "[0.2, 0.3, 0.4]")
    sh = SimulationHistogram(trajectory_path=PATH_OUTPUT_PT, trajectory_name="NH3_H2O_0000", full_grid=my_fg,
                             energies="None",
                             second_molecule_selection="bynum 5:7")
    assignments = sh.get_quaternion_assignments()
    np.set_printoptions(threshold=sys.maxsize)
    print(assignments.astype(int)[1::8])
    print(np.unique(assignments.astype(int), return_counts=True))

