import numpy as np

from MDAnalysis import AtomGroup

from molgri.molecules.pts import Pseudotrajectory
from molgri.scripts.set_up_io import copy_examples, freshly_create_all_folders
from molgri.space.fullgrid import FullGrid
from molgri.space.utils import normalise_vectors
from molgri.io import OneMoleculeReader


def same_distance(mol1: AtomGroup, mol2: AtomGroup):
    """Check that two molecules have the same distance from COM to origin."""
    dist1 = np.linalg.norm(mol1.center_of_mass())
    dist2 = np.linalg.norm(mol2.center_of_mass())
    assert np.isclose(dist1, dist2, atol=1e-5)


def same_body_orientation(mol1: AtomGroup, mol2: AtomGroup):
    """If you subtract COM, molecules should look the same"""
    com1 = mol1.center_of_mass()
    com2 = mol2.center_of_mass()
    centered_pos1 = mol1.positions - np.tile(com1, (mol1.positions.shape[0],1))
    centered_pos2 = mol2.positions - np.tile(com2, (mol1.positions.shape[0], 1))
    assert np.allclose(centered_pos2-centered_pos1, 0, atol=1e-5)

def same_origin_orientation(mol1: AtomGroup, mol2: AtomGroup):
    """Check that two molecules have COM on the same vector from the origin (have the same orientation with respect
    to origin)"""
    unit_pos1 = normalise_vectors(mol1.center_of_mass())
    unit_pos2 = normalise_vectors(mol2.center_of_mass())
    assert np.allclose(unit_pos1, unit_pos2, atol=1e-5)


def test_pt_len():
    """
    Test that PTs are of correct length.
    """
    TEST_M1 = ["input/H2O.gro", "input/H2O.gro", "input/H2O.gro"]
    TEST_M2 = ["input/H2O.gro", "input/NH3.gro", "input/NH3.gro"]
    NUM_ORIENTATIONS = [17, 5, 1]
    NUM_DIRECTIONS = [17, 18, 9]
    TGRID = ["linspace(1, 5, 10)", "[1,2,3,4,5,6,7,8,9,10]", "range(0.1,0.2,0.01)"]

    for m1, m2, no,nd, tgrid in zip(TEST_M1, TEST_M2, NUM_ORIENTATIONS, NUM_DIRECTIONS, TGRID):
        grid = FullGrid(f"{no}", f"{nd}", tgrid).get_full_grid_as_array()
        pt = Pseudotrajectory(OneMoleculeReader(m1).get_molecule(), OneMoleculeReader(m2).get_molecule(), grid)
        max_i=0
        for i, _ in pt.generate_pseudotrajectory():
            max_i=i
        max_i += 1
        # +1 because indices are from 0 to N-1 and we are looking for N
        assert max_i==no*nd*10, f"Last index is {max_i} and not {no*nd*10}"


def test_pt_translations():
    """
    Test that if there are no rotations of any kind, translation occurs at correct distances in the z-direction.
    """
    # on a zero rotation grids
    grid = FullGrid(f"1", f"1", "range(1, 5, 0.5)")
    grid_array = grid.get_full_grid_as_array()
    distances = grid.get_position_grid().get_radii()
    pt = Pseudotrajectory(OneMoleculeReader("input/H2O.gro").get_molecule(), OneMoleculeReader(
        "input/H2O.gro").get_molecule(), grid_array)
    for i, frame in pt.generate_pseudotrajectory():
        # first three atoms should be basically at zero
        assert np.allclose(frame.select_atoms("bynum 1:3").center_of_mass(), 0)
        # other three atoms (0, 0, distance)
        com2 = frame.select_atoms("bynum 4:6").center_of_mass()
        assert np.isclose(com2[0], 0)
        assert np.isclose(com2[1], 0)
        assert np.isclose(com2[2], distances[i])

def test_pt_rotations_origin():
    """
    Test that if there is no body rotation grid,
        1) distances to COM are equal to those prescribed by translational grid
        2) angles between vector to COM and vectors to individual atoms stay constant (water stays the shape of water)
    """
    num_rot = 12
    num_trans = 2
    m1 = OneMoleculeReader("input/H2O.gro").get_molecule()

    grid = FullGrid(f"1", f"{num_rot}", "[0.5,0.7]")
    grid_array = grid.get_full_grid_as_array()
    distances = grid.get_position_grid().get_radii()
    pt = Pseudotrajectory(m1, m1, grid_array)

    # initial angles
    vec_com_0 = m1.atoms.center_of_mass()
    vec_atom1_0 = m1.atoms[0].position - vec_com_0
    vec_atom2_0 = m1.atoms[1].position - vec_com_0
    vec_atom3_0 = m1.atoms[2].position - vec_com_0

    for i, frame in pt.generate_pseudotrajectory():
        # first three atoms should be basically at zero
        assert np.allclose(frame.select_atoms("bynum 1:3").center_of_mass(), 0)
        # other three atoms (0, 0, distance)
        com2 = frame.select_atoms("bynum 4:6").center_of_mass()
        m2 = frame.select_atoms("bynum 4:6")
        dist = np.linalg.norm(com2)
        if i < num_rot:  # first n_o elements will be at same radius
            assert np.isclose(dist, distances[0], atol=1e-5)
        else:  # even indices, lower orbit
            assert np.isclose(dist, distances[1], atol=1e-5)
        # all should have same body orientation
        same_body_orientation(m1.atoms, m2)

        vec_com = m2.center_of_mass()
        vec_atom1 = m2.atoms[0].position - vec_com
        vec_atom2 = m2.atoms[1].position - vec_com
        vec_atom3 = m2.atoms[2].position - vec_com

        assert np.allclose(vec_atom1_0, vec_atom1)
        assert np.allclose(vec_atom2_0, vec_atom2)
        assert np.allclose(vec_atom3_0, vec_atom3)



def test_pt_rotations_body():
    """
    test that if there is no direction grid,
        1) distances to COM are equal to those prescribed by translational grid
        2) COM only moves in the z-direction, x and y coordinates of COM stay at 0
    """
    num_rot = 4
    num_trans = 3
    # assert every uneven structure has distance 1 and every even one distance 2

    m1 = OneMoleculeReader("input/H2O.gro").get_molecule()
    m2 = OneMoleculeReader("input/NH3.gro").get_molecule()

    grid = FullGrid(f"{num_rot}", f"1", "[1, 2, 3]")
    grid_array = grid.get_full_grid_as_array()
    distances = grid.get_position_grid().get_radii()
    pt = Pseudotrajectory(m1, m2, grid_array)

    for i, frame in pt.generate_pseudotrajectory():
        amoniak = frame.select_atoms("bynum 4:7")

        dist = np.linalg.norm(amoniak.center_of_mass())
        if i < num_rot:  # the first n_o*n_b points at same (smallest) radius
            assert np.isclose(dist, distances[0], atol=1e-5), f"{dist}!={distances[0]}"
        elif i < 2*num_rot:
            assert np.isclose(dist, distances[1], atol=1e-5), f"{dist}!={distances[1]}"
        else:
            assert np.isclose(dist, distances[2], atol=1e-5), f"Frame {i}: {dist}!={distances[2]}"
        # x and y coordinates of COM of molecule 2 stay 0, z coordinate is same as distance
        com_2 = amoniak.center_of_mass()
        assert np.isclose(com_2[0], 0, atol=1e-5)
        assert np.isclose(com_2[1], 0, atol=1e-5)
        assert np.isclose(com_2[2], dist, atol=1e-5)


def test_order_of_operations():
    """
    If you have n_o rotational orientations, n_b body rotations and n_t translations, the first n_b elements
    should have the same position/COM, the first n_t also (?) and this pattern
    continuous to repeat.

    - 0, 1, ... n_b-1 have the exact same position, n_b, n_b+1 ... 2*n_b-1 have the exact same position...
    - 0, n_b, 2*n_b, 3*n_b ... have the same orientation, 1, n_b+1, 2*n_b+1, 3*n_b+1 ... have the same orientation
    """
    n_b = 4
    n_o = 8
    n_t = 3

    m1 = OneMoleculeReader("input/H2O.gro").get_molecule()
    m2 = OneMoleculeReader("input/NH3.gro").get_molecule()

    grid = FullGrid(f"{n_b}", f"{n_o}", "[1, 2, 3]")
    grid_array = grid.get_full_grid_as_array()
    distances = grid.get_position_grid().get_radii()
    pt = Pseudotrajectory(m1, m2, grid_array)

    second_molecule_frames = []

    max_i = 0
    for i, frame in pt.generate_pseudotrajectory():
        max_i = i
        second_molecule_frames.append(frame.select_atoms("bynum 4:7"))


    assert max_i+1 == n_b * n_o * n_t

    # result[k*N_b:(k+1)*N_b] for integer k: same vector from origin (same position
    for k in range(n_t*n_o):
        first_one = second_molecule_frames[k*n_b]
        for o in range(k*n_b, (k+1)*n_b):
            second_one = second_molecule_frames[o]
            same_distance(first_one, second_one)
            same_origin_orientation(first_one, second_one)


    # result[k::N_b] for integer k is the same body orientation
    for k in range(n_b):
        first_one = second_molecule_frames[k]
        for o in range(k, n_b * n_o * n_t, n_b):
            second_one = second_molecule_frames[o]
            same_body_orientation(first_one, second_one)


if __name__ == "__main__":
    freshly_create_all_folders()
    copy_examples()
    test_pt_len()
    test_pt_translations()
    test_pt_rotations_origin()
    test_pt_rotations_body()
    test_order_of_operations()
