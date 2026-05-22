"""
When we have periodic molecules as part of the system (molecule 1) we face a particular challenge: the simulation cell
must be pretty large so that molecule 2 doesn't interact with itself, but the unit cell is much smaller.

We want our grid to only fill out the unit cell. Actually, for simplicity we want the smallest cuboid unit cell
- a rhomboid might form a smaller unit cell but then we would need to adapt our translation grid. Therefore, in the
function find_cuboid_cell we automatically search which combination of unit vectors form a cuboid repeating
cell.
"""

from itertools import product

import plotly.graph_objects as go
from MDAnalysis import Universe
from ase.geometry import cellpar_to_cell
from ase.io import read
from numpy.typing import NDArray
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from ase.build import make_supercell
from scipy.spatial import cKDTree



def is_orthorhombic(lattice: NDArray, ang_tol_deg=1e-2) -> bool:
    """
    This is a simple check that a lattice defined by a 3x3 lattice where each row is a basis vector is (
    approximately) orthogonal. The test could be simpler using linear algebra, but because we want to control the
    numerical error, we calculate the three angles directly.

    Args:
        lattice (NDAarray): lattice array of hape (3,3), each row is a basis vector
        ang_tol_deg (float): allowing the angles between vectors to not be perfect 90° angles

    Returns:
        True if the provided lattice is close enough to a orthogonal vector set, else False
    """
    a, b, c = lattice[0], lattice[1], lattice[2]
    def angle(u, v):
        cu = np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v))
        cu = np.clip(cu, -1.0, 1.0)
        return np.degrees(np.arccos(cu))
    alpha = angle(b, c)   # should be ~90
    beta  = angle(a, c)
    gamma = angle(a, b)
    return (abs(alpha - 90.0) <= ang_tol_deg and
            abs(beta  - 90.0) <= ang_tol_deg and
            abs(gamma - 90.0) <= ang_tol_deg)

def is_periodic(positions: NDArray, candidate_lattice: NDArray, pos_tol: float = 0.01) -> bool:
    """
    Check if 'candidate_lattice' yields a repeating cell for the given Cartesian positions.

    How this is done:
    1) from Cartesian coordinates r_i we calculate fractional coordinates f_i between 0 and 1
    2) then we confirm that all initial positions r_i can be written as

        r_i = (f_i + n) @ candidate lattice where n=[-1, 0, 1]

    Args:
        positions (NDArray): the positions of atoms
        candidate_lattice (NDArray): the lattice, a 3x3 matrix where each row is a lattice vector
        pos_tol (float): tolerance for not perfectly symmetric structure

    Returns:
        True if this candidate lattice is infinitely repeating
    """
    inverse_lattice = np.linalg.inv(candidate_lattice.T)
    # where the atom is relative to the candidate lattice
    coordinates_fraction_of_cell = np.dot(positions, inverse_lattice)
    coordinates_fraction_of_1 = coordinates_fraction_of_cell - np.floor(coordinates_fraction_of_cell)

    # the expanded set of points are original points shifted to each possible boundary cell (like the image of
    # periodic boundary conditions) meaning that each point is now repeated 3x3x3=27 times
    shifts = list(product([-1,0,1], repeat=3))
    expanded = []
    for s in shifts:
        shift = np.array(s)
        f = coordinates_fraction_of_1 + shift
        expanded.append(np.dot(f, candidate_lattice))
    expanded = np.vstack(expanded)

    # now we should find that one of the coordinates in the expanded set exactly matches the original position
    # (distances all close to zero) - if the lattice is indeed periodic
    tree = cKDTree(expanded)
    dists, idx = tree.query(positions, k=1)
    return bool(np.all(dists <= pos_tol))

def find_cuboid_cell(ase_cell: list, ase_positions: NDArray, numerator_options =(-2,-1,0,1,2),
                          denominator_options = (1,2,4), ang_tol_deg=1e-2, pos_tol=1e-3) -> NDArray:
    """
    Find the lattice vectors that describe the smallest cuboid cell that describe the periodic molecule provided in
    positions.

    How this works:
    1) we have the current lattice provided as ase_cell, so we have three vectors a, b, c that describe a unit cell
    2) now we build new candidate lattice vectors as linear combinations of original lattice vectors of a form

        v = xa + yb + zc where x,y,z in have a form p/q where p and q are small positive or negative integers

    intuitively, this means that the new lattice vectors can be formed e.g. as 1a + 1/2b. This creates candidate
    lattice vectors that might be orthogonal and might preserve long-range order
    3) now we test the orthogonality and repetition conditions
    4) of all remaining lattice vectors we select the one with the smallest volume (because of course multiples of
    this unit cell are also unit vectors)


    Args:
        ase_cell (list): cell information, side lengths and angles like [0,0,0,90,90,90]
        ase_positions (list): cartesian positions of atoms in the cell
        numerator_options (tuple): options for the selection of p in equation above
        denominator_options (tuple): options for the selection of q in equation above
        ang_tol_deg (float): tolerance for not perfectly orthogonal cells
        pos_tol (float): tolerance for not perfectly symmetric structure

    Returns:
        an array in which each row is a lattice vector of a cuboid cell
    """
    L = np.array( ase_cell, dtype=float)
    results = []

    # coefficient space for one vector is product of p/q choices for each of 3 components
    # we iterate over coefficient matrices of shape (3,3) where each element = p/q
    # to limit combinatorics, we can first build coefficient tuples for single vector and then combine for three vectors
    coeff_vectors = []
    for q in denominator_options:
        for p in product(numerator_options, repeat=3):
            coeff = np.array(p, dtype=float) / float(q)
            if np.allclose(coeff, 0.0):
                continue
            coeff_vectors.append(coeff)

    coeff_vectors = np.array(coeff_vectors)

    # now build candidate lattice vectors as linear combinations coeff @ L where L is current lattice length
    # to reduce loops, only consider combinations where the 3 coefficient vectors are linearly independent
    for v1c in coeff_vectors:
        v1 = v1c @ L
        for v2c in coeff_vectors:
            v2 = v2c @ L
            # quick linear independence check
            if np.linalg.matrix_rank(np.vstack([v1c, v2c])) < 2:
                continue
            for v3c in coeff_vectors:
                # quick linear independence check
                if np.linalg.matrix_rank(np.vstack([v1c, v2c, v3c])) < 3:
                    continue
                v3 = v3c @ L
                candidate = np.vstack([v1, v2, v3])
                # ensure oriented properly: nonzero volume
                det = np.linalg.det(candidate)
                if abs(det) < 1e-8:
                    continue
                # ensure that the cell is orthorhombic
                if not is_orthorhombic(candidate, ang_tol_deg=ang_tol_deg):
                    continue
                # We may want to ensure vectors are right-handed; if det < 0, flip one
                if det < 0:
                    candidate[2] = -candidate[2]
                    det = -det
                if not is_periodic(np.array(ase_positions), candidate, pos_tol=pos_tol):
                    continue
                vol = abs(det)
                results.append({
                    'lattice': candidate,
                    'coeff_matrix': np.vstack([v1c, v2c, v3c]),
                    'volume': vol
                })
                # sort so that the first element is the smallest-volume one
                results = sorted(results, key=lambda r: (r['volume'], np.sum(np.abs(r['coeff_matrix']))))
    return results[0]["lattice"]


def find_primitive_cell(path_structure: str, precision: float = 0.001) -> Structure:
    """
    Find the smallest (but not necessarily cuboid) unit cell of structure at path_structure.

    Args:
        path_structure (str): where to find the structure file (eg .gro file) of a periodic molecule
        precision (float): allowed deviations when acessing symmetry

    Returns:
        a Structure instance that provides the lattice of primitive cell
    """
    atoms = read(path_structure)
    atoms.set_cell(cellpar_to_cell([14.5086, 14.5086, 14.5086, 60, 60, 60]))
    atoms.set_pbc(True)
    structure = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(structure, symprec=precision, angle_tolerance=5)
    prim = sga.get_primitive_standard_structure()
    return prim


def get_x_y_grid_inputs(structure_path: str, num_x_points: int, num_y_points: int) -> tuple[list, list]:
    """
    This is a high-level function to automatically set x and y grid parameters for a repeating molecular structure
    provided in structure_path. The grid starts at (0, 0) and extends to the smallest repeatable cuboid cell.

    How this is done:
    1) we first determine the (often non-cuboid) primitive cell
    2) we then tile the primitive cell in xy direction to get a supercell
    3) based on the supercell we find a cuboid lattice with the smallest volume that is repeatable

    When you are using this for a new system, you MUST TEST that the assignment of cuboid cell is correct by
    visualizing the cuboid cell on the structure - see molecular_visualizations.smk

    Args:
        structure_path (str): path to the structure file (gro file)
        num_x_points (int): number of grid points in x direction
        num_y_points (int): number of grid points in y direction

    Returns:
        [[x_min, x_max, x_step], [y_min, y_max, y_step]]
    """
    max_x, max_y, max_z = get_cuboid_cell_side_lengths(structure_path)
    return [0, max_x, num_x_points], [0, max_y, num_y_points]

def get_cuboid_cell_side_lengths(structure_path: str):
    primitive_structure = find_primitive_cell(structure_path)
    primitive_atoms = AseAtomsAdaptor.get_atoms(primitive_structure)

    supercell_atoms = make_supercell(primitive_atoms, np.diag([2,2,1]))

    cuboid_lattice = find_cuboid_cell(supercell_atoms.get_cell(), np.array(supercell_atoms.get_positions()),
                                                numerator_options=(-1,0,1), denominator_options=(1, 2, 4))

    Lx = float(cuboid_lattice[0][0])
    Ly = float(cuboid_lattice[1][1])
    Lz = float(cuboid_lattice[2][2])
    return np.array([Lx, Ly, Lz])

def wrap_to_cuboid_cell(origin: NDArray, side_lengths: NDArray, coordinates: NDArray,
                        wrap_only_xy: bool = False) -> NDArray:
    """
    A simulation is done in a larger simulation cell. However, since the cuboid cell is infinitely repeating,
    each coordinate outside the cuboid cell can be "wrapped" to a corresponding position within the cell. This
    function returns the coordinates wrapped to the cuboid cell. This is done using the modulo operator.

    Args:
        origin (NDArray): an array of shape (3,) describing where the unit cell starts - typically (0,0,0)
        side_lengths (NDArray): an array of shape (3,) describing the length of the cell in each direction
        coordinates (NDArray): an array of shape (N_points, 3) of coordinates that might be inside or outside the
            cuboid cell
        wrap_only_xy (bool): if True, leave the z-coordinate as-is and only wrap x and y coordinates

    Returns:
        an array of shape (N_points, 3) where each point of input coordinates is transformed so that it lies at a
        corresponding position in the cuboid cell
    """
    if wrap_only_xy:
        wrapped_2D = origin[:2] + np.mod(coordinates[:,:2] - origin[:2], side_lengths[:2])
        return np.column_stack((wrapped_2D, coordinates[:, 2]))
    return origin + np.mod(coordinates - origin, side_lengths)

def wrap_multiple_atoms_to_cuboid_cell(com: NDArray, coordinates: NDArray, side_lengths: NDArray,
                                       wrap_only_xy: bool = False) -> NDArray:
    """
    Similar to wrap_to_cuboid_cell, but the coordinates are not independent, all belong to the same construct,
    e.g. same molecule. Therefore, they should be wrapped together. This method ensures that the center of mass lies
    inside the cuboid cell, but some of the individual coordinates might lie slightly outside of it.

    Args:
        com (NDArray): an array of shape (3,) for center of mass of given coordinates
        coordinates (NDArray): an array of shape (N_points, 3) for coordinates that should be wrapped
        side_lengths (NDArray): an array of shape (3,) describing the length of the cell in each direction
        wrap_only_xy (bool): if True, leave the z-coordinate as-is and only wrap x and y coordinates

    Returns:
        an array of shape (N_points, 3) where each point of input coordinates is transformed by the same vector so
        that COM lies inside the cuboid cell
    """
    vector_com_to_atom = coordinates - com
    com_wrapped = wrap_to_cuboid_cell(np.zeros((3,)), side_lengths, com[np.newaxis, :], wrap_only_xy)
    if wrap_only_xy:
        return np.column_stack((vector_com_to_atom[:, :2] + com_wrapped[:, :2], coordinates[:, 2]))
    else:
        return vector_com_to_atom + com_wrapped

if __name__ == "__main__":
    """
    Here we have some very useful plotting to visualize the process:
    1) we first determine the (often non-cuboid) primitive cell
    2) we then tile the primitive cell in xy direction to get a supercell
    3) based on the supercell we find a cuboid lattice with the smallest volume that is repeatable
    """
    from molgri.images.plotting import draw_structure, draw_unit_cell

    my_path = "/home/hanaz63/2026_molgri/nobackup/MOF_num_AR/40_40_40/pseudosimulation/"

    structure = f"{my_path}structure.xyz"
    structure1 = f"{my_path}molecule1.xyz"
    trajectory = f"{my_path}trajectory.xyz"


    primitive_structure = find_primitive_cell(structure1)
    primitive_atoms = AseAtomsAdaptor.get_atoms(primitive_structure)
    primitive_cell = primitive_atoms.cell.cellpar()
    print("primitive_atoms ", primitive_atoms)


    supercell_atoms = make_supercell(primitive_atoms, np.diag([3,3,3]))
    supercell_structure = AseAtomsAdaptor.get_structure(supercell_atoms)
    supercell_cell = supercell_atoms.get_cell().cellpar()

    # cuboid_unit_structure = find_cuboid_cell(supercell_atoms.get_cell(), supercell_atoms.get_positions(),
    #                                                    numerator_options=(-1,0,1), denominator_options=(1, 3)) #, 2, 4
    #
    # print(cuboid_unit_structure)


    fig = go.Figure()
    print("matrix ", primitive_structure.lattice)
    draw_unit_cell(fig, primitive_cell[:3], primitive_cell[3:], color="blue")
    draw_unit_cell(fig, supercell_cell[:3], supercell_cell[3:], color="red")
    draw_unit_cell(fig, np.array([14.5086, 12.564816173346905, 11.84622229] ), np.array([90,90,90]),
                   color="green", show=True)
    #draw_unit_cell(fig, cuboid_unit_structure.diagonal(), color="green")

    #draw_structure(fig, structure1, show=True)