import os

import numpy as np

from molgri.writers import PtIOManager, directory2full_pt, converter_gro_dir_gro_file_names, full_pt2directory
from molgri.parsers import PtParser, ParsedMolecule
from molgri.scripts.set_up_io import freshly_create_all_folders, copy_examples
from molgri.utils import angle_between_vectors, normalise_vectors
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
    manager = PtIOManager(m_path, m_path, f"ico_{num_rotations}", f"randomE_{num_rotations}", "linspace(1, 5, 10)")
    manager.construct_pt()
    end_index = manager.pt.current_frame
    num_translations = manager.full_grid.t_grid.get_N_trans()
    assert end_index == num_translations * num_rotations * num_rotations
    parser = PtParser(f"{PATH_INPUT_BASEGRO}{m_path}", f"{PATH_INPUT_BASEGRO}{m_path}",
                      f"{PATH_OUTPUT_PT}{manager.determine_pt_name()}.gro",
                      f"{PATH_OUTPUT_PT}{manager.determine_pt_name()}.xtc")
    assert len(parser.universe.trajectory) == end_index
    assert parser.c_num == parser.r_num == 3
    # 2nd example
    num_body = 3
    num_origin = 18
    m1_path = f"H2O.gro"
    m2_path = f"NH3.gro"
    manager = PtIOManager(m1_path, m2_path, f"cube4D_{num_origin}", f"randomQ_{num_body}", "range(1, 5, 0.5)")
    manager.construct_pt()
    end_index = manager.pt.current_frame
    num_translations = manager.full_grid.t_grid.get_N_trans()
    assert end_index == num_translations * num_body * num_origin
    parser = PtParser(f"{PATH_INPUT_BASEGRO}{m1_path}", f"{PATH_INPUT_BASEGRO}{m2_path}",
                      f"{PATH_OUTPUT_PT}{manager.determine_pt_name()}.gro",
                      f"{PATH_OUTPUT_PT}{manager.determine_pt_name()}.xtc")
    assert len(parser.universe.trajectory) == end_index
    assert parser.r_num == 4
    assert parser.c_num == 3
    # 3rd example
    num_origin = 9
    num_body = 1
    m1_path = f"H2O.gro"
    m2_path = f"NH3.gro"
    manager = PtIOManager(m1_path, m2_path, f"cube4D_{num_origin}", f"zero", "range(1, 5, 0.5)")
    manager.construct_pt()
    end_index = manager.pt.current_frame
    num_translations = manager.full_grid.t_grid.get_N_trans()
    assert end_index == num_translations * num_body * num_origin
    parser = PtParser(f"{PATH_INPUT_BASEGRO}{m1_path}",
                      f"{PATH_INPUT_BASEGRO}{m2_path}",
                      f"{PATH_OUTPUT_PT}{manager.determine_pt_name()}.gro",
                      f"{PATH_OUTPUT_PT}{manager.determine_pt_name()}.xtc")
    assert len(parser.universe.trajectory) == end_index
    assert parser.r_num == 4
    assert parser.c_num == 3


def test_pt_translations():
    """
    Test that if there are no rotations of any kind, translation occurs at correct distances in the z-direction.
    """
    # on a zero rotation grids
    m_path = f"H2O.gro"
    manager = PtIOManager(m_path, m_path, "zero", "zero", "range(1, 5, 0.5)")
    manager.construct_pt()
    distances = manager.full_grid.t_grid.get_trans_grid()
    file_name = manager.determine_pt_name()
    # center of mass of the second molecule moves in z direction

    traj_parser = PtParser(f"{PATH_INPUT_BASEGRO}{m_path}",
                           f"{PATH_INPUT_BASEGRO}{m_path}",
                           f"{PATH_OUTPUT_PT}{file_name}.gro",
                           f"{PATH_OUTPUT_PT}{file_name}.xtc")
    for frame_i, frame_molecules in enumerate(traj_parser.generate_frame_as_molecule()):
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
    manager = PtIOManager(m_path, m_path, f"ico_{num_rot}", "zero", "[1, 2]")
    distances = manager.full_grid.t_grid.get_trans_grid()
    manager.construct_pt()
    file_name = manager.determine_pt_name()
    len_traj = manager.pt.current_frame
    assert len_traj == num_trans*num_rot
    # assert every uneven structure has distance 1 and every even one distance 2
    traj_parser = PtParser(f"{PATH_INPUT_BASEGRO}{m_path}",
                           f"{PATH_INPUT_BASEGRO}{m_path}",
                           f"{PATH_OUTPUT_PT}{file_name}.gro",
                           f"{PATH_OUTPUT_PT}{file_name}.xtc")
    for frame_i, frame_molecules in enumerate(traj_parser.generate_frame_as_molecule()):
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
        assert np.isclose(angle_between_vectors(vec_com, vec_atom1), 1.386, atol=0.03)
        assert np.isclose(angle_between_vectors(vec_com, vec_atom2), 1.227, atol=0.03)
        assert np.isclose(angle_between_vectors(vec_com, vec_atom3), 2.152, atol=0.03)


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
    manager = PtIOManager(m1_path, m2_path, f"zero", f"cube4D_{num_rot}", "[1, 2, 3]")
    distances = manager.full_grid.t_grid.get_trans_grid()
    manager.construct_pt_and_time()
    file_name = manager.determine_pt_name()
    len_traj = manager.pt.current_frame
    assert len_traj == num_trans*num_rot
    # assert every uneven structure has distance 1 and every even one distance 2
    traj_parser = PtParser(f"{PATH_INPUT_BASEGRO}{m1_path}",
                           f"{PATH_INPUT_BASEGRO}{m2_path}",
                           f"{PATH_OUTPUT_PT}{file_name}.gro",
                           f"{PATH_OUTPUT_PT}{file_name}.xtc")
    for frame_i, frame_molecules in enumerate(traj_parser.generate_frame_as_molecule()):
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

    manager = PtIOManager(m1_path, m2_path, f"ico_{n_o}", f"randomQ_{n_b}", "[1, 2, 3]")
    manager.construct_pt()
    file_name = manager.determine_pt_name()
    len_traj = manager.pt.current_frame
    assert len_traj == n_b * n_o * n_t
    # assert every uneven structure has distance 1 and every even one distance 2
    traj_parser = PtParser(f"{PATH_INPUT_BASEGRO}{m1_path}",
                           f"{PATH_INPUT_BASEGRO}{m2_path}",
                           f"{PATH_OUTPUT_PT}{file_name}.gro",
                           f"{PATH_OUTPUT_PT}{file_name}.xtc")
    m2s = []
    for frame_i, frame_molecules in enumerate(traj_parser.generate_frame_as_molecule()):
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


def test_frames_in_directory():
    n_b = 4
    n_o = 2
    n_t = 3
    manager = PtIOManager("H2O", "NH3", f"randomE_{n_o}", f"systemE_{n_b}", "[1, 2, 3]")
    manager.construct_pt(as_dir=True)
    file_name = manager.determine_pt_name()
    base_p, fn, fp, dp = converter_gro_dir_gro_file_names(pt_directory_path=f"{PATH_OUTPUT_PT}{file_name}",
                                                          extension="xtc")
    directory2full_pt(dp)
    new_name = base_p + "joined_" + fn + ".xtc"
    os.rename(fp, new_name)
    filelist = [f for f in os.listdir(f"{dp}") if f.endswith(".xtc")]
    assert len(filelist) == n_b*n_o*n_t, "Not correct number of .xtc files in a directory."
    # compare contents of individual files mit the single all-frame PT
    manager = PtIOManager("H2O", "NH3", f"randomE_{n_o}", f"systemE_{n_b}", "[1, 2, 3]")
    manager.construct_pt(as_dir=False)
    # check that a directory joined version is the same as the normal one
    with open(new_name, "rb") as f1:
        with open(fp, "rb") as f2:
            l1 = f1.readlines()
            l2 = f2.readlines()
        assert np.all(l1 == l2)


def test_directory_combined_to_pt():
    # check that a full directory can be split
    n_b = 7
    n_o = 1
    manager = PtIOManager("H2O", "NH3", f"cube4D_{n_o}", f"cube4D_{n_b}", "[1, 2, 3]")
    manager.construct_pt(as_dir=False)
    file_name = manager.determine_pt_name()
    base_p, fn, fp, dp = converter_gro_dir_gro_file_names(pt_file_path=f"{PATH_OUTPUT_PT}{file_name}.xtc")
    full_pt2directory(fp, structure_path=f"{base_p}{fn}.gro")
    new_dir_name = base_p + "split_" + fn + "/"
    if os.path.exists(new_dir_name):
        # delete contents if folder already exist
        filelist = [f for f in os.listdir(new_dir_name)]
        for f in filelist:
            os.remove(os.path.join(new_dir_name, f))
        os.rmdir(new_dir_name)
    os.rename(dp, new_dir_name)
    # the non-split version / created during PT creation
    manager2 = PtIOManager("H2O", "NH3", f"cube4D_{n_o}", f"cube4D_{n_b}", "[1, 2, 3]")
    manager2.construct_pt(as_dir=True)
    filelist1 = [f for f in os.listdir(f"{new_dir_name}") if f.endswith(".xtc")]
    filelist2 = [f for f in os.listdir(f"{dp}") if f.endswith(".xtc")]
    filelist1.sort(key=lambda x: int(x.split(".")[0]))
    filelist2.sort(key=lambda x: int(x.split(".")[0]))
    for file_name1, file_name2 in zip(filelist1, filelist2):
        path1 = new_dir_name + file_name1
        path2 = dp + file_name2
        with open(path1, "r") as f1:
            lines1 = f1.readlines()
        with open(path2, "r") as f2:
            lines2 = f2.readlines()
        assert np.all(lines1 == lines2)


if __name__ == '__main__':
    freshly_create_all_folders()
    copy_examples()
    # test_pt_len()
    # test_pt_translations()
    test_pt_rotations_origin()
    # test_pt_rotations_body()
    # test_order_of_operations()
    # test_frames_in_directory()
    # test_directory_combined_to_pt()
