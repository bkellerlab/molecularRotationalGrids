import numpy as np

from molgri.molecules.writers import PtIOManager
from molgri.scripts.set_up_io import copy_examples, freshly_create_all_folders
from molgri.molecules.parsers import PtParser, ParsedMolecule
from molgri.space.utils import angle_between_vectors, normalise_vectors
from molgri.paths import PATH_OUTPUT_PT, PATH_INPUT_BASEGRO


def same_distance(mol1: ParsedMolecule, mol2: ParsedMolecule):
    """Check that two molecules have the same distance from COM to origin."""
    dist1 = np.linalg.norm(mol1.get_center_of_mass())
    dist2 = np.linalg.norm(mol2.get_center_of_mass())
    assert np.isclose(dist1, dist2, atol=1e-3)


def same_body_orientation(mol1: ParsedMolecule, mol2: ParsedMolecule):
    """Check that two molecules (that should have the same atoms) have the same internal body orientation."""
    assert np.shape(mol1.atoms) == np.shape(mol2.atoms)
    assert np.all(mol1.atom_types == mol2.atom_types)
    for atom1, atom2 in zip(mol1.atoms, mol2.atoms):
        vec_com1 = mol1.get_center_of_mass()
        vec_atom1 = atom1.position - vec_com1
        angle1 = angle_between_vectors(vec_com1, vec_atom1)
        vec_com2 = mol2.get_center_of_mass()
        vec_atom2 = atom2.position - vec_com2
        angle2 = angle_between_vectors(vec_com2, vec_atom2)
        assert np.isclose(angle1, angle2, atol=0.05)


def same_origin_orientation(mol1: ParsedMolecule, mol2: ParsedMolecule):
    """Check that two molecules have COM on the same vector from the origin (have the same orientation with respect
    to origin)"""
    unit_pos1 = normalise_vectors(mol1.get_center_of_mass())
    unit_pos2 = normalise_vectors(mol2.get_center_of_mass())
    assert np.allclose(unit_pos1, unit_pos2, atol=1e-3)


def test_pt_len():
    """
    Test that PTs are of correct length in various cases, including zero rotational grids. Check this in three
    ways, so that the product of n_o*n_b*n_t is equal to:
     - index returned by pt.generate_pseudotrajectory()
     - last t=value comment in the PT file minus one
     - length of the PT file when you consider how many lines each frame takes
    """
    # origin rot grid = body rot grid
    num_rotations = 55
    m_path = f"H2O.gro"
    manager = PtIOManager(m_path, m_path, f"{num_rotations}", f"{num_rotations}", "linspace(1, 5, 10)")
    manager.construct_pt()
    end_index = manager.pt.current_frame
    num_translations = manager.full_grid.t_grid.get_N_trans()
    assert end_index == num_translations * num_rotations * num_rotations
    parser = PtParser(f"{PATH_INPUT_BASEGRO}{m_path}", f"{PATH_INPUT_BASEGRO}{m_path}", manager.output_paths[1],
                      manager.output_paths[0])
    assert len(parser.universe.trajectory) == end_index
    assert parser.c_num == parser.r_num == 3
    # 2nd example
    num_body = 3
    num_origin = 18
    m1_path = f"H2O.gro"
    m2_path = f"NH3.gro"
    manager = PtIOManager(m1_path, m2_path, f"{num_origin}", f"{num_body}", "range(1, 5, 0.5)")
    manager.construct_pt()
    end_index = manager.pt.current_frame
    num_translations = manager.full_grid.t_grid.get_N_trans()
    assert end_index == num_translations * num_body * num_origin
    parser = PtParser(f"{PATH_INPUT_BASEGRO}{m1_path}", f"{PATH_INPUT_BASEGRO}{m2_path}",
                      manager.output_paths[1],
                      manager.output_paths[0])
    assert len(parser.universe.trajectory) == end_index
    assert parser.r_num == 4
    assert parser.c_num == 3
    # 3rd example
    num_origin = 9
    num_body = 1
    m1_path = f"H2O.gro"
    m2_path = f"NH3.gro"
    manager = PtIOManager(m1_path, m2_path, f"{num_origin}", f"{num_body}", "range(1, 5, 0.5)")
    manager.construct_pt()
    end_index = manager.pt.current_frame
    num_translations = manager.full_grid.t_grid.get_N_trans()
    assert end_index == num_translations * num_body * num_origin
    parser = PtParser(f"{PATH_INPUT_BASEGRO}{m1_path}",
                      f"{PATH_INPUT_BASEGRO}{m2_path}",
                      manager.output_paths[1],
                      manager.output_paths[0])
    assert len(parser.universe.trajectory) == end_index
    assert parser.r_num == 4
    assert parser.c_num == 3


def test_pt_translations():
    """
    Test that if there are no rotations of any kind, translation occurs at correct distances in the z-direction.
    """
    # on a zero rotation grids
    m_path = f"H2O.gro"
    manager = PtIOManager(m_path, m_path, "1", "1", "range(1, 5, 0.5)")
    manager.construct_pt()
    distances = manager.full_grid.t_grid.get_trans_grid()
    file_name = manager.determine_pt_name()
    # center of mass of the second molecule moves in z direction

    traj_parser = PtParser(f"{PATH_INPUT_BASEGRO}{m_path}",
                           f"{PATH_INPUT_BASEGRO}{m_path}",
                           manager.output_paths[1],
                           manager.output_paths[0])
    for frame_i, frame_molecules in enumerate(traj_parser.generate_frame_as_double_molecule()):
        # distance of COM of second molecule to origin
        molecule1, molecule2 = frame_molecules
        # molecule 2 should be translated along the z coordinate
        assert np.allclose(molecule2.get_center_of_mass()[2], distances[frame_i], atol=1e-3)
        # x and y coordinates should not change
        assert np.allclose(molecule2.get_center_of_mass()[0], molecule1.get_center_of_mass()[0], atol=1e-3)
        assert np.allclose(molecule2.get_center_of_mass()[1], molecule1.get_center_of_mass()[1], atol=1e-3)


def test_pt_rotations_origin():
    """
    Test that if there is no body rotation grid,
        1) distances to COM are equal to those prescribed by translational grid
        2) angles between vector to COM and vectors to individual atoms stay constant
    """
    num_rot = 12
    num_trans = 2
    m_path = f"H2O.gro"
    manager = PtIOManager(m_path, m_path, f"{num_rot}", "1", "[1, 2]")
    distances = manager.full_grid.t_grid.get_trans_grid()
    manager.construct_pt()
    file_name = manager.determine_pt_name()
    len_traj = manager.pt.current_frame
    assert len_traj == num_trans*num_rot
    # assert every uneven structure has distance 1 and every even one distance 2
    traj_parser = PtParser(f"{PATH_INPUT_BASEGRO}{m_path}",
                           f"{PATH_INPUT_BASEGRO}{m_path}",
                           manager.output_paths[1],
                           manager.output_paths[0])
    # initial angles
    m1, m2 = next(traj_parser.generate_frame_as_double_molecule())
    vec_com_0 = m2.get_center_of_mass()
    vec_atom1_0 = m2.atoms[0].position - vec_com_0
    vec_atom2_0 = m2.atoms[1].position - vec_com_0
    vec_atom3_0 = m2.atoms[2].position - vec_com_0
    angle_start_1 = angle_between_vectors(vec_com_0, vec_atom1_0)
    angle_start_2 = angle_between_vectors(vec_com_0, vec_atom2_0)
    angle_start_3 = angle_between_vectors(vec_com_0, vec_atom3_0)
    # should stay the same during a trajectory
    for frame_i, frame_molecules in enumerate(traj_parser.generate_frame_as_double_molecule()):
        # distance of COM of second molecule to origin
        m1, m2 = frame_molecules
        dist = np.linalg.norm(m2.get_center_of_mass())
        if frame_i % 2 == 1:  # odd indices, higher orbit
            assert np.isclose(dist, distances[1], atol=1e-3)
        else:   # even indices, lower orbit
            assert np.isclose(dist, distances[0], atol=1e-3)
        # calculate angles from atom positions to coordinate axes
        vec_com = m2.get_center_of_mass()
        vec_atom1 = m2.atoms[0].position - vec_com
        vec_atom2 = m2.atoms[1].position - vec_com
        vec_atom3 = m2.atoms[2].position - vec_com
        assert np.isclose(angle_between_vectors(vec_com, vec_atom1), angle_start_1, atol=0.03)
        assert np.isclose(angle_between_vectors(vec_com, vec_atom2), angle_start_2, atol=0.03)
        assert np.isclose(angle_between_vectors(vec_com, vec_atom3), angle_start_3, atol=0.03)


def test_pt_rotations_body():
    """
    test that if there is no orientational grid,
        1) distances to COM are equal to those prescribed by translational grid
        2) COM only moves in the z-direction, x and y coordinates of COM stay at 0
    """
    num_rot = 4
    num_trans = 3
    # assert every uneven structure has distance 1 and every even one distance 2
    m1_path = f"H2O.gro"
    m2_path = f"NH3.gro"
    manager = PtIOManager(m1_path, m2_path, f"1", f"{num_rot}", "[1, 2, 3]")
    distances = manager.full_grid.t_grid.get_trans_grid()
    manager.construct_pt_and_time()
    file_name = manager.determine_pt_name()
    len_traj = manager.pt.current_frame
    assert len_traj == num_trans*num_rot
    # assert every uneven structure has distance 1 and every even one distance 2
    traj_parser = PtParser(f"{PATH_INPUT_BASEGRO}{m1_path}",
                           f"{PATH_INPUT_BASEGRO}{m2_path}",
                           manager.output_paths[1],
                           manager.output_paths[0])
    for frame_i, frame_molecules in enumerate(traj_parser.generate_frame_as_double_molecule()):
        m1, m2 = frame_molecules
        dist = np.linalg.norm(m2.get_center_of_mass())
        if frame_i % 3 == 0:
            assert np.isclose(dist, distances[0], atol=1e-3)
        elif frame_i % 3 == 1:
            assert np.isclose(dist, distances[1], atol=1e-3)
        else:  # even indices, lower orbit
            assert np.isclose(dist, distances[2], atol=1e-3)
        # x and y coordinates of COM of molecule 2 stay 0, z coordinate is same as distance
        com_2 = m2.get_center_of_mass()
        assert np.isclose(com_2[0], 0, atol=1e-3)
        assert np.isclose(com_2[1], 0, atol=1e-3)
        assert np.isclose(com_2[2], dist, atol=1e-3)


def test_order_of_operations():
    """
    If you have n_o rotational orientations, n_b body rotations and n_t translations, the first n_b*n_t elements
    should have the same space orientation, the first n_t also the same body orientation and this pattern
    continuous to repeat.
    """
    n_b = 4
    n_o = 8
    n_t = 3
    m1_path = f"H2O.gro"
    m2_path = f"NH3.gro"

    manager = PtIOManager(m1_path, m2_path, f"{n_o}", f"{n_b}", "[1, 2, 3]")
    manager.construct_pt()
    file_name = manager.determine_pt_name()
    len_traj = manager.pt.current_frame
    assert len_traj == n_b * n_o * n_t
    # assert every uneven structure has distance 1 and every even one distance 2
    traj_parser = PtParser(f"{PATH_INPUT_BASEGRO}{m1_path}",
                           f"{PATH_INPUT_BASEGRO}{m2_path}",
                           manager.output_paths[1],
                           manager.output_paths[0])
    m2s = []
    for frame_i, frame_molecules in enumerate(traj_parser.generate_frame_as_double_molecule()):
        m1, m2 = frame_molecules
        m2s.append(m2)
    # each batch of n_b*n_t elements should have the same space orientation
    for o in range(0, len_traj, n_b*n_t):
        for i in range(o, o+n_b*n_t):
            mol2_ts_i = m2s[i]
            for j in range(i+1, o+n_b*n_t):
                mol2_ts_j = m2s[j]
                same_origin_orientation(mol2_ts_i, mol2_ts_j)
    # each batch of n_t also the same body orientation
    for o in range(0, len_traj, n_t):
        for i in range(o, o+n_t):
            mol2_ts_i = m2s[i]
            for j in range(i+1, o+n_t):
                mol2_ts_j = m2s[j]
                same_body_orientation(mol2_ts_i, mol2_ts_j)


if __name__ == "__main__":
    freshly_create_all_folders()
    copy_examples()
    test_pt_len()
    test_pt_translations()
    test_pt_rotations_origin()
    test_pt_rotations_body()
    test_order_of_operations()
