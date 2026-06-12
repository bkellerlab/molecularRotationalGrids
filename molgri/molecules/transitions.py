"""
In this file we are building Markov State Models and Sqare Root-Approximations to determine slow processes. We also
perform eigendecomposition of these matrices.
"""

from typing import Optional, Sequence, Tuple, Any

import pandas as pd
from numpy.typing import NDArray
import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import coo_array, csr_array, diags_array, dok_array, vstack, hstack

from scipy.constants import k as kB, N_A
from sklearn.neighbors import KernelDensity
from petsc4py import PETSc
from slepc4py import SLEPc

from molgri.molecules.rate_merger import expand_eigenvector_to_full_length
from molgri.utils.arrays import k_argmax_in_array, k_argmin_in_array



def scipy_to_petsc(A):
    A = A.tocsr()
    indptr = A.indptr.astype('int32')
    indices = A.indices.astype('int32')

    petsc_A = PETSc.Mat().createAIJ(
        size=A.shape,
        csr=(indptr, indices, A.data)
    )
    petsc_A.assemble()

    return petsc_A

def decompose_with_petsc(A, sigma=0.0, num_eigenvectors=12):
    A = scipy_to_petsc(A)
    E = SLEPc.EPS().create()
    E.setOperators(A)

    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)  # general non-Hermitian
    E.setDimensions(nev=num_eigenvectors)  # number of eigenpairs
    E.setTolerances(tol=1e-10, max_it=200000)

    # target eigenvalues near zero
    E.setTarget(sigma)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

    E.setFromOptions()
    E.solve()

    nconv = E.getConverged()
    print('nconv', nconv)
    vals = []
    vecs = []

    vr, vi = A.getVecs()  # real/imag parts

    for i in range(nconv):
        eigval = E.getEigenpair(i, vr, vi)

        # convert PETSc vector → NumPy
        v = vr.getArray().copy()

        vals.append(eigval)
        vecs.append(v)

    vals = np.array(vals)
    vecs = np.column_stack(vecs)
    return vals, vecs

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
    Subsample the sequence so that only each len_window-th element remains and then similarly return pairs of
    elements.

    In this case we throw more data away but we avoid correlated data.

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
    As an input we have am assigned trajectory - an array as long as the trajectory but each element is an index (of a
    gridpoint that best describes the structure at this frame index). As an output we create a MSM transition matrix.
    """

    def __init__(self, assigned_trajectory: NDArray, total_num_cells: int):
        self.assigned_trajectory = assigned_trajectory
        self.total_num_cells = total_num_cells

    def get_one_tau_transition_matrix(self, tau: int, noncorrelated_windows: bool)-> csr_array:
        """
        This is the main method, it constructs a MSM for a specific tau ("time jump"). We count transitions using
        detailed balance and normalize them.

        Args:
            tau (int): the size of the window
            noncorrelated_windows (bool): if True use non-correlated windows, else correlated

        Returns:
            a sparse matrix of shape (N_gridpoints, N_gridpoints) where element (i, j)=(j, i) tells us how likely
            transition between state i and j is at the timescale of tau
        """
        sparse_count_matrix = dok_array((self.total_num_cells, self.total_num_cells))
        # save the number of transitions between cell with index i and cell with index j
        # count_per_cell = {(i, j): 0 for i in range(self.num_cells) for j in range(self.num_cells)}
        if noncorrelated_windows:
            window_cell = noncorr_window(self.assigned_trajectory, int(tau))
        else:
            window_cell = window(self.assigned_trajectory, int(tau))

        for cell_slice in window_cell:
            # in case we only have one last element available, we skip
            if len(cell_slice)<2:
                pass
            else:
                el1, el2 = cell_slice
                sparse_count_matrix[el1, el2] += 1
                # enforce detailed balance
                sparse_count_matrix[el2, el1] += 1
        sparse_count_matrix = sparse_count_matrix.tocsr()
        sums = sparse_count_matrix.sum(axis=1)
        # to avoid dividing by zero
        sums[sums == 0] = 1
        # now dividing with counts (actually multiplying with inverse)
        diagonal_values = np.reciprocal(sums)
        diagonal_matrix = diags_array(diagonal_values, format='csr')
        # Left multiply the CSR matrix with the diagonal matrix
        result = diagonal_matrix.dot(sparse_count_matrix)
        return result


class SQRA:

    """
    Square-root approximation is an alternative to a Markov model. We need a lot more information about the grid:
    distances, surfaces, volumes - but only one energy evaluation per cell
    """

    def __init__(self, energies: NDArray, volumes: csr_array, distances: csr_array, surfaces: csr_array, T: float):
        self.energies = energies
        self.volumes = volumes
        self.distances = distances
        self.surfaces = surfaces
        self.T = T
        self.rate_matrix = None

    def add_bulk(self, flow_to_bulk: float, surfaces_to_bulk: NDArray, energy_kJ_mol_bulk: float, volumes_to_bulk: NDArray):
        sum_surfaces = np.sum(surfaces_to_bulk)
        V_bulk = 6563 #62563.28#299.4# 25563.287333333334
        # energy ratio is cca 1 because the interaction energy already basically zero)
        energy_ratios = np.exp((energy_kJ_mol_bulk-self.energies) * 1000 / (2 * kB * N_A * self.T))
        #print(flow_to_bulk/surfaces_to_bulk)
        inv_energy_ratios = np.exp((self.energies-energy_kJ_mol_bulk) * 1000 / (2 * kB * N_A * self.T))
        final_flow = np.where(surfaces_to_bulk != 0, flow_to_bulk*surfaces_to_bulk/sum_surfaces * energy_ratios, 0.0)
        new_row = csr_array(final_flow)
        print(new_row)
        self.rate_matrix = vstack([self.rate_matrix, new_row])
        final_flow_column = np.where(surfaces_to_bulk != 0, flow_to_bulk*surfaces_to_bulk /sum_surfaces *
                                     inv_energy_ratios, 0.0)
        final_flow_column = np.hstack((final_flow_column, -np.sum(final_flow)))
        final_flow_column = final_flow_column.reshape((-1, 1))
        new_column = csr_array(final_flow_column)
        print(new_column)
        self.rate_matrix = hstack([self.rate_matrix, new_column])
        return self.rate_matrix


    def get_rate_matrix(self, diffusion_matrix: csr_array, capping_factor: float) -> csr_array:
        """
        This is the method that gets from cell properties (energies, volumes) and adjacency properties (distances,
        surfaces) to the full rate matrix.

        This method will cause overflow warnings - but don't worry we're dealing with them in delete_rows_columns method

        Args:
            D (float): diffusion constant, currently just a float TODO must become an adjacency property
            T (float): the temperature of the simulation

        Returns:
            a sparse array of rates of shape (N_gridpoints, N_gridpoints)
        """
        # for sqra demand that each energy corresponds to exactly one cell
        #assert len(self.energies) == len(self.volumes), f"{len(self.energies)} != {len(self.volumes)}"
        # you cannot multiply or divide directly in a coo format
        # using a higher-precision dtype is not useful, since we take exponentials of huge numbers - always overflow
        rate_matrix = self.surfaces
        print("surfaces ", pd.DataFrame(self.surfaces.data).describe())
        rate_matrix = rate_matrix.tocoo()
        rate_matrix.data *= diffusion_matrix.tocoo().data
        print("diffusion_matrix ", pd.DataFrame(diffusion_matrix.data).describe())
        rate_matrix.data /= self.distances.tocoo().data
        # Divide every row of transition_matrix with the corresponding volume
        rate_matrix.data /= self.volumes.tocoo().data #[rate_matrix.row]
        # multiply with sqrt(pi_j/pi_i) = e**((V_i-V_j)*1000/(2*k_B*N_A*T))
        # gromacs uses kJ/mol as energy unit, boltzmann constant is J/K
        energy_differences = self.energies[rate_matrix.row] - self.energies[rate_matrix.col]
        energy_differences[np.isnan(energy_differences)] = np.inf
        # import pandas as pd
        # print(pd.DataFrame(energy_differences).describe())

        if capping_factor!="None":
            energy_differences = np.where(energy_differences < float(capping_factor), energy_differences, float(capping_factor))

        pi_exponent = energy_differences * 1000 / (2 * kB * N_A * self.T)
        print("pi_exponent ", pd.DataFrame(pi_exponent).describe())
        rate_matrix.data *= np.exp(pi_exponent)
        print("rate matrix before normalizing ", pd.DataFrame(rate_matrix.data).describe())
        #print(pd.DataFrame(rate_matrix.data).describe())

        # normalize sum of rows
        rate_matrix.setdiag(0)
        sums = rate_matrix.sum(axis=1)
        sum_diag = diags_array(-sums, format="csr")
        all_together = rate_matrix + sum_diag
        self.rate_matrix = all_together
        print("rate matrix after normalizing ", pd.DataFrame(rate_matrix.data).describe())
        return all_together



class DecompositionTool:

    """
    Just a simple wrapper to perform decomposition and assure the eigenvectors and/or eigenvalues are not complex.
    """

    def __init__(self, matrix_to_decompose: NDArray | csr_array | coo_array, kept_indices: NDArray, total_length: int):
        """

        Args:
            matrix_to_decompose (NDArray | csr_array | coo_array): a single matrix to be decomposed
            kept_indices (NDArray): the kept indices in case the provided matrix was reduced with some row-column
                pairs deleted (see rate_merger.py for more information)
            total_length (int): the final length to which the eigenvectors will be expanded
        """
        self.matrix_to_decompose = matrix_to_decompose
        self.kept_indices = kept_indices
        self.total_length = total_length

        # scale
        self.scale = 1 #np.abs(self.matrix_to_decompose.data).max()
        self.matrix_to_decompose = self.matrix_to_decompose / self.scale

    def decompose_msm(self) -> tuple:
        """
        Decomposition with settings suitable for transition matrices (expected first eigenvalue 1 and all others
        positive).

        Returns:
            (eigenvalues, eigenvectors) where eigenvalues is an array of shape (12,) and eigenvectors an array of
            shape (total_len, 12)
        """
        return self.get_decomposition(tol=1e-12, maxiter=100000, which="LR", sigma=1.0)

    def decompose_sqra(self, sigma, tolerance, **kwargs) -> tuple:
        """
        Decomposition with settings suitable for transition matrices (expected first eigenvalue 0).

        Returns:
            (eigenvalues, eigenvectors) where eigenvalues is an array of shape (12,) and eigenvectors an array of
            shape (total_len, 12)
        """
        eigenvalues, eigenvectors = self.get_decomposition(tol=tolerance, maxiter=10000000, which="SR", sigma=sigma,
                                                           **kwargs)
        # now divide into three categories: sum +-1, sum 0, sum other
        indices_sum_to_0 = []
        indices_sum_to_other = []
        for i, expanded_eigenvector in enumerate(eigenvectors.T):
            print(i, np.abs(np.sum(expanded_eigenvector)))
            if np.isclose(np.sum(expanded_eigenvector), 0, atol=1e-3, rtol=1e-3):
                indices_sum_to_0.append(i)
            else:
                indices_sum_to_other.append(i)
        # return the three categories
        first_tuple = (eigenvalues[indices_sum_to_0], eigenvectors[:, indices_sum_to_0])
        third_tuple = (eigenvalues[indices_sum_to_other], eigenvectors[:, indices_sum_to_other])
        return first_tuple, third_tuple

    def get_decomposition(self, tol: float, maxiter: int, which: str, sigma: Optional[float], k: int = 15) -> tuple:
        """
        The function to decompose matrices. It wraps the scipy decompose and makes sure:
        - the output is not given as complex numbers
        - the eigenvectors are sorted by corresponding eigenvalues
        - the eigenvectors are expanded to total_length

        Args:
            tol (float): the tolerance for eigendecomposition
            maxiter (int): max number of cycles for decomposition
            which (str): "SR", "LR", "SM" or "LM", defines the type of eigenvalues we look for,
            for more info  see documentation of scipy.eigs
            sigma (float or None): search for eigenvalues close to this value, for more info see documentation of
            scipy.eigs

        Returns:
            (eigenvalues, eigenvectors) where eigenvalues is an array of shape (12,) and eigenvectors an array of
            shape (total_len, 12)
        """
        # two options for transpose: large, sparse matrices need specialized methods but small ones perform better
        # with full eigendecomposition
        eigenval, eigenvec = decompose_with_petsc(self.matrix_to_decompose.T, sigma, num_eigenvectors=k)
        # if self.matrix_to_decompose.shape[0] > 20000:
        #     eigenval, eigenvec = eigs(self.matrix_to_decompose.T, k=k, tol=tol, maxiter=maxiter, which=which,
        #                               sigma=sigma)
        # else:
        #     eigenval, eigenvec = eig(self.matrix_to_decompose.toarray(), left=True, right=False)
        # if imaginary eigenvectors or eigenvalues, raise error
        if not np.allclose(eigenvec.imag.max(), 0, rtol=1e-5, atol=1e-3) or not np.allclose(eigenval.imag.max(), 0,
                                                                                            rtol=1e-5, atol=1e-3):
            print(f"Complex values for eigenvectors and/or eigenvalues: {eigenvec.imag.max()}{eigenval.imag.max()}")
        eigenvec = eigenvec.real
        eigenval = eigenval.real

        # sort eigenvectors according to their eigenvalues
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]

        expanded_eigenvectors = []
        # expand to full length
        for eigenvector in eigenvec.T:
            expanded_eigenvector = expand_eigenvector_to_full_length(eigenvector, self.kept_indices, self.total_length)

            expanded_eigenvectors.append(expanded_eigenvector)
        expanded_eigenvectors = np.array(expanded_eigenvectors)

        final_eigenvalues = eigenval * self.scale
        final_eigenvectors = expanded_eigenvectors
        final_eigenvectors = final_eigenvectors.T
        return final_eigenvalues, final_eigenvectors


def kde_valley_cutoffs(data: NDArray, bandwidth: str |float = "scott", grid_size: int = 2000, peak_prominence: float = 0.01) -> tuple:
    """
    In a 1D data set find "peaks", areas of high point density and "valleys", areas separating one peak from the next
    with low point density. To do this, density is modelled with KernelDensity.

    Args:
        data (NDArray): an array of shape (N_datapoints,) for which the density will be modelled
        bandwidth (str |float): either a metho describing bandwith determination ('scott', 'silverman') or numeric bandwidth
        grid_size (int): number of evaluation points for KDE
        peak_prominence (float): the smaller the number, the more peaks will be found

    Returns:
        a tuple (peaks, valleys) where the two elements are lists of floats; peaks describes the positions of highest
        densities and valleys the positions of lowest densities
    """
    x = np.asarray(data).reshape(-1, 1)

    # --- bandwidth selection ---
    if isinstance(bandwidth, str):
        std = np.std(x)
        n = len(x)
        if bandwidth == "scott":
            bw = std * (n ** (-1 / 5))
        elif bandwidth == "silverman":
            bw = 1.06 * std * (n ** (-1 / 5))
        else:
            raise ValueError("bandwidth must be 'scott', 'silverman', or float")
    else:
        bw = float(bandwidth)

    # --- KDE fit ---
    kde = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde.fit(x)

    grid = np.linspace(x.min(), x.max(), grid_size)
    log_density = kde.score_samples(grid.reshape(-1, 1))
    density = np.exp(log_density)

    peaks, _ = find_peaks(density, prominence=peak_prominence)

    # if len(peaks) < 2:
    #     raise ValueError("Less than 2 peaks detected; distribution may be unimodal.")

    valleys, _ = find_peaks(-density)
    return grid[peaks], grid[valleys]

def auto_determine_eigenvector_extremes(one_eigenvector: NDArray,  N_extremes_to_plot: int | str) -> tuple:
    """
    The idea of this function: we might not want to plot excatly 5 or 50 structures with most positive contributions,
    but automatically select the number of structures that form a cluster around the most positive and most negative
    values.

    Args:
        one_eigenvector (NDArray): an array of shape (N_gridpoints,) that gives you the contribution to this
            eigenvector for each gridpoint (cell)
        N_extremes_to_plot (int | str): if an integer, returns the N smallest and largest contributions
            if the word "auto", modells the distribution of points and select the N smallest and N largest that
            correspond to clusters

    Returns:
        a tuple of two arrays (indices_most_positive, indices_most_negative)
        the content of the arrays are always the grid indices that have the most positive or negative contributions
        to the eigenvector
        if N_extremes_to_plot is an integer, both arrays will have the length N_extremes_to_plot
        if N_extremes_to_plot is "auto", the arrays might have different lengths
    """
    # should be automatically determined
    if N_extremes_to_plot == "auto":
        above_upper_limit = np.zeros(100)
        below_lower_limit = np.zeros(100)

        peak_prominence = 0.01
        max_num_cycles = 5
        cycle_num = 0

        while (len(above_upper_limit) > 10 or len(below_lower_limit) > 10) and cycle_num < max_num_cycles:
            my_peaks, my_valleys = kde_valley_cutoffs(one_eigenvector, peak_prominence=peak_prominence)
            upper_limit = np.max(my_valleys)
            lower_limit = np.min(my_valleys)

            above_upper_limit = np.where(one_eigenvector >= upper_limit)[0]
            below_lower_limit = np.where(one_eigenvector <= lower_limit)[0]
            peak_prominence = peak_prominence / 5
            cycle_num += 1
        if len(above_upper_limit) > 10:
            above_upper_limit = above_upper_limit[:10]
        if len(below_lower_limit) > 10:
            below_lower_limit = below_lower_limit[:10]
        return above_upper_limit, below_lower_limit
    # if we already know how many structures to plot
    else:
        N_extremes_to_plot = int(N_extremes_to_plot)
        return k_argmax_in_array(one_eigenvector, N_extremes_to_plot), k_argmin_in_array(one_eigenvector,N_extremes_to_plot)

