import os

import numpy as np

from molgri.bodies import Molecule
from molgri.writers import TwoMoleculeGroWriter, PtWriter, directory2full_pt, full_pt2directory, \
    converter_gro_dir_gro_file_names
from molgri.grids import IcoGrid, Cube4DGrid, ZeroGrid, FullGrid
from molgri.parsers import TranslationParser, MultiframeGroParser
from molgri.scripts.set_up_io import freshly_create_all_folders, copy_examples
from molgri.utils import angle_between_vectors, normalise_vectors
from molgri.paths import PATH_OUTPUT_PT


def same_distance(mol1: Molecule, mol2: Molecule):
    """Check that two molecules have the same distance from COM to origin."""
    dist1 = np.linalg.norm(mol1.position)
    dist2 = np.linalg.norm(mol2.position)
    assert np.isclose(dist1, dist2, atol=1e-3)


def same_body_orientation(mol1: Molecule, mol2: Molecule):
    """Check that two molecules (that should have the same atoms) have the same internal body orientation."""
    assert np.shape(mol1.atoms) == np.shape(mol2.atoms)
    assert np.all([atom1.element == atom2.element for atom1, atom2 in zip(mol1.atoms, mol2.atoms)])
    for atom1, atom2 in zip(mol1.atoms, mol2.atoms):
        vec_com1 = mol1.position
        vec_atom1 = atom1.position - vec_com1
        angle1 = angle_between_vectors(vec_com1, vec_atom1)
        vec_com2 = mol2.position
        vec_atom2 = atom2.position - vec_com2
        angle2 = angle_between_vectors(vec_com2, vec_atom2)
        assert np.isclose(angle1, angle2, atol=0.05)


def same_origin_orientation(mol1: Molecule, mol2: Molecule):
    """Check that two molecules have COM on the same vector from the origin (have the same orientation with respect
    to origin)"""
    unit_pos1 = normalise_vectors(mol1.position)
    unit_pos2 = normalise_vectors(mol2.position)
    assert np.allclose(unit_pos1, unit_pos2, atol=1e-3)


def test_two_molecule_gro():
    pt = TwoMoleculeGroWriter("H2O", "H2O", 0.5)
    pt.write_full_pt_gro()
    with open(pt.file_name, "r") as f:
        lines = f.readlines()
        num_atoms = 3 + 3
        num_oth_lines = 3
        assert len(lines) == (num_atoms + num_oth_lines), "Wrong number of lines in .gro file"


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
    rot_grid = IcoGrid(num_rotations)
    trans_grid = TranslationParser("linspace(1, 5, 10)")
    num_translations = trans_grid.get_N_trans()
    full_grid = FullGrid(b_grid=rot_grid, o_grid=rot_grid, t_grid=trans_grid)
    writer = PtWriter("H2O", "H2O", full_grid)
    writer.write_full_pt_gro()
    file_name = writer.file_name
    end_index = writer.pt.current_frame
    assert end_index == num_translations*num_rotations*num_rotations
    with open(file_name, "r") as f:
        lines = f.readlines()
        num_atoms = 3 + 3
        num_oth_lines = 3
        assert len(lines) == (num_atoms + num_oth_lines) * end_index, "Wrong number of lines in .gro file"
        last_t_comment = int(lines[-num_oth_lines-num_atoms].split("=")[-1].strip())  # like c_num=3, r_num=3, t=1
        assert last_t_comment == end_index - 1, "Comment of the last frame not equal to num of frames -1."
    # origin rot grid =/= body rot grid
    num_body = 3
    body_grid = IcoGrid(num_body)
    num_origin = 18
    origin_grid = Cube4DGrid(num_origin)
    trans_grid = TranslationParser("range(1, 5, 0.5)")
    num_translations = trans_grid.get_N_trans()
    full_grid = FullGrid(b_grid=body_grid, o_grid=origin_grid, t_grid=trans_grid)
    writer = PtWriter("H2O", "NH3", full_grid=full_grid)
    writer.write_full_pt_gro()
    file_name = writer.file_name
    end_index = writer.pt.current_frame
    assert end_index == num_translations * num_body * num_origin
    with open(file_name, "r") as f:
        lines = f.readlines()
        num_atoms = 4 + 3
        num_oth_lines = 3
        assert len(lines) == (num_atoms + num_oth_lines) * end_index, "Wrong number of lines in .gro file"
        last_t_comment = int(lines[-num_oth_lines-num_atoms].split("=")[-1].strip())  # like c_num=3, r_num=4, t=1
        assert last_t_comment == end_index - 1, "Comment of the last frame not equal to num of frames -1."
    # body grid is null (only_origin)
    num_origin = 9
    origin_grid = Cube4DGrid(num_origin)
    trans_grid = TranslationParser("range(1, 5, 0.5)")
    num_translations = trans_grid.get_N_trans()
    full_grid = FullGrid(b_grid=ZeroGrid(), o_grid=origin_grid, t_grid=trans_grid)
    writer = PtWriter("H2O", "NH3", full_grid)
    writer.write_full_pt_gro()
    file_name = writer.file_name
    end_index = writer.pt.current_frame
    assert end_index == num_translations * num_origin
    with open(file_name, "r") as f:
        lines = f.readlines()
        num_atoms = 4 + 3
        num_oth_lines = 3
        assert len(lines) == (num_atoms + num_oth_lines) * end_index, "Wrong number of lines in .gro file"
        last_t_comment = int(lines[-num_oth_lines-num_atoms].split("=")[-1].strip())  # like c_num=3, r_num=4, t=1
        assert last_t_comment == end_index - 1, "Comment of the last frame not equal to num of frames -1."


def test_pt_translations():
    """
    Test that if there are no rotations of any kind, translation occurs at correct distances in the z-direction.
    """
    # on a zero rotation grids
    trans_grid = TranslationParser("range(1, 5, 0.5)")
    full_grid = FullGrid(t_grid=trans_grid, b_grid=ZeroGrid(), o_grid=ZeroGrid())
    distances = trans_grid.get_trans_grid()
    writer = PtWriter("H2O", "H2O", full_grid)
    writer.write_full_pt_gro()
    file_name = writer.file_name
    # center of mass of the second molecule moves in z direction
    ts = MultiframeGroParser(file_name, is_pt=True).timesteps
    for i, t in enumerate(ts):
        molecule1 = t.molecule_set.all_objects[0]
        molecule2 = t.molecule_set.all_objects[1]
        # molecule 2 should be translated along the z coordinate
        assert np.allclose(molecule2.position[2], distances[i], atol=1e-3)
        # x and y coordinates should not change
        assert np.allclose(molecule2.position[0], molecule1.position[0], atol=1e-3)
        assert np.allclose(molecule2.position[1], molecule1.position[1], atol=1e-3)
    assert np.isclose(ts[-1].t, len(distances)-1)  # -1 because we start numbering timeframes from zero


def test_pt_rotations_origin():
    """
    Test that if there is no body rotation grid,
        1) distances to COM are equal to those prescribed by translational grid
        2) angles between vector to COM and vectors to individual atoms stay constant
    """
    num_rot = 12
    ico_grid = IcoGrid(num_rot)
    num_trans = 2
    trans_grid = TranslationParser("[1, 2]")
    distances = trans_grid.get_trans_grid()
    full_grid = FullGrid(t_grid=trans_grid, b_grid=ZeroGrid(), o_grid=ico_grid)
    writer = PtWriter("H2O", "H2O", full_grid)
    writer.write_full_pt_gro()
    file_name = writer.file_name
    len_traj = writer.pt.current_frame
    assert len_traj == num_trans*num_rot
    # assert every uneven structure has distance 1 and every even one distance 2
    ts = MultiframeGroParser(file_name, parse_atoms=True, is_pt=True).timesteps
    for frame_i in range(num_rot*num_trans):
        # distance of COM of second molecule to origin
        dist = np.linalg.norm(ts[frame_i].molecule_set.all_objects[1].position)
        if frame_i % 2 == 1:  # odd indices, higher orbit
            assert np.isclose(dist, distances[1], atol=1e-3)
        else:   # even indices, lower orbit
            assert np.isclose(dist, distances[0], atol=1e-3)
        # calculate angles from atom positions to coordinate axes
        vec_com = ts[frame_i].molecule_set.all_objects[1].position
        vec_atom1 = ts[frame_i].molecule_set.all_objects[1].atoms[0].position - vec_com
        vec_atom2 = ts[frame_i].molecule_set.all_objects[1].atoms[1].position - vec_com
        vec_atom3 = ts[frame_i].molecule_set.all_objects[1].atoms[2].position - vec_com
        assert np.isclose(angle_between_vectors(vec_com, vec_atom1), 0.9, atol=0.03)
        assert np.isclose(angle_between_vectors(vec_com, vec_atom2), 3.08, atol=0.03)
        assert np.isclose(angle_between_vectors(vec_com, vec_atom3), 1.26, atol=0.03)


def test_pt_rotations_body():
    """
    test that if there is no orientational grid,
        1) distances to COM are equal to those prescribed by translational grid
        2) COM only moves in the z-direction, x and y coordinates of COM stay at 0
    """
    num_rot = 4
    ico_grid = IcoGrid(num_rot)
    num_trans = 3
    trans_grid = TranslationParser("[1, 2, 3]")
    distances = trans_grid.get_trans_grid()
    full_grid = FullGrid(t_grid=trans_grid, b_grid=ico_grid, o_grid=ZeroGrid())
    writer = PtWriter("H2O", "NH3", full_grid)
    writer.write_full_pt_gro(measure_time=True)
    file_name = writer.file_name
    len_traj = writer.pt.current_frame
    assert len_traj == num_trans*num_rot
    # assert every uneven structure has distance 1 and every even one distance 2
    ts = MultiframeGroParser(file_name, parse_atoms=True, is_pt=True).timesteps
    for frame_i in range(num_rot*num_trans):
        # distance of COM of second molecule to origin
        dist = np.linalg.norm(ts[frame_i].molecule_set.all_objects[1].position)
        if frame_i % 3 == 0:
            assert np.isclose(dist, distances[0], atol=1e-3)
        elif frame_i % 3 == 1:
            assert np.isclose(dist, distances[1], atol=1e-3)
        else:   # even indices, lower orbit
            assert np.isclose(dist, distances[2], atol=1e-3)
        # x and y coordinates of COM of molecule 2 stay 0, z coordinate is same as distance
        com_2 = ts[frame_i].molecule_set.all_objects[1].position
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
    grid_b = IcoGrid(n_b)
    n_o = 8
    grid_o = IcoGrid(n_o)
    n_t = 3
    trans_grid = TranslationParser("[1, 2, 3]")
    full_grid = FullGrid(t_grid=trans_grid, b_grid=grid_b, o_grid=grid_o)
    writer = PtWriter("H2O", "NH3", full_grid)
    writer.write_full_pt_gro()
    file_name = writer.file_name
    len_traj = writer.pt.current_frame
    ts = MultiframeGroParser(file_name, parse_atoms=True, is_pt=True).timesteps
    # each batch of n_b*n_t elements should have the same space orientation
    for o in range(0, len_traj, n_b*n_t):
        for i in range(o, o+n_b*n_t):
            mol2_ts_i = ts[i].molecule_set.all_objects[1]
            for j in range(i+1, o+n_b*n_t):
                mol2_ts_j = ts[j].molecule_set.all_objects[1]
                same_origin_orientation(mol2_ts_i, mol2_ts_j)
    # each batch of n_t also the same body orientation
    for o in range(0, len_traj, n_t):
        for i in range(o, o+n_t):
            mol2_ts_i = ts[i].molecule_set.all_objects[1]
            for j in range(i+1, o+n_t):
                mol2_ts_j = ts[j].molecule_set.all_objects[1]
                same_body_orientation(mol2_ts_i, mol2_ts_j)


def test_frames_in_directory():
    n_b = 4
    grid_b = IcoGrid(n_b)
    n_o = 2
    grid_o = IcoGrid(n_o)
    n_t = 3
    trans_grid = TranslationParser("[1, 2, 3]")
    full_grid = FullGrid(t_grid=trans_grid, b_grid=grid_b, o_grid=grid_o)
    writer = PtWriter("H2O", "NH3", full_grid)
    writer.write_frames_in_directory()
    base_p, fn, fp, dp = converter_gro_dir_gro_file_names(pt_file_path=writer.file_name)
    directory2full_pt(dp)
    new_name = base_p + "joined_" + fn + ".gro"
    os.rename(fp, new_name)
    filelist = [f for f in os.listdir(f"{dp}") if f.endswith(".gro")]
    assert len(filelist) == n_b*n_o*n_t, "Not correct number of .gro files in a directory."
    # compare contents of individual files mit the single all-frame PT
    writer2 = PtWriter("H2O", "NH3", full_grid)
    writer2.write_full_pt_gro()
    # check that a directory joined version is the same as the normal one
    with open(new_name, "r") as f1:
        with open(fp, "r") as f2:
            l1 = f1.readlines()
            l2 = f2.readlines()
        assert np.all(l1 == l2)


def test_directory_combined_to_pt():
    # check that a full directory can be split
    n_b = 7
    grid_b = IcoGrid(n_b)
    n_o = 1
    grid_o = IcoGrid(n_o)
    n_t = 3
    trans_grid = TranslationParser("[1, 2, 3]")
    full_grid = FullGrid(t_grid=trans_grid, b_grid=grid_b, o_grid=grid_o)
    writer = PtWriter("H2O", "NH3", full_grid)
    writer.write_full_pt_gro()
    base_p, fn, fp, dp = converter_gro_dir_gro_file_names(pt_file_path=writer.file_name)
    full_pt2directory(fp)
    new_dir_name = base_p + "split_" + fn + "/"
    if os.path.exists(new_dir_name):
        # delete contents if folder already exist
        filelist = [f for f in os.listdir(new_dir_name) if f.endswith(".gro")]
        for f in filelist:
            os.remove(os.path.join(new_dir_name, f))
        os.rmdir(new_dir_name)
    os.rename(dp, new_dir_name)
    # the non-split version / created during PT creation
    writer2 = PtWriter("H2O", "NH3", full_grid)
    writer2.write_frames_in_directory()
    filelist1 = [f for f in os.listdir(f"{new_dir_name}") if f.endswith(".gro")]
    filelist2 = [f for f in os.listdir(f"{dp}") if f.endswith(".gro")]
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
