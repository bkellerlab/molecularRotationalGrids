import numpy as np

from molgri.bodies import Molecule
from molgri.pts import Pseudotrajectory
from molgri.grids import IcoGrid, Cube4DGrid, ZeroGrid
from molgri.parsers import TranslationParser, MultiframeGroParser
from molgri.scripts.set_up_io import freshly_create_all_folders, copy_examples
from molgri.paths import PATH_OUTPUT_PT
from molgri.utils import angle_between_vectors, normalise_vectors


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
    pt = Pseudotrajectory("H2O", "H2O", rot_grid_origin=rot_grid, trans_grid=trans_grid, rot_grid_body=rot_grid)
    file_name = f"{PATH_OUTPUT_PT}{pt.pt_name}.gro"
    end_index = pt.generate_pseudotrajectory()
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
    pt = Pseudotrajectory("H2O", "NH3", rot_grid_origin=origin_grid, trans_grid=trans_grid, rot_grid_body=body_grid)
    file_name = f"{PATH_OUTPUT_PT}{pt.pt_name}.gro"
    end_index = pt.generate_pseudotrajectory()
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
    pt = Pseudotrajectory("H2O", "NH3", rot_grid_origin=origin_grid, trans_grid=trans_grid, rot_grid_body=None)
    file_name = f"{PATH_OUTPUT_PT}{pt.pt_name}.gro"
    end_index = pt.generate_pseudotrajectory()
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
    zg = ZeroGrid()
    trans_grid = TranslationParser("range(1, 5, 0.5)")
    distances = trans_grid.get_trans_grid()
    pt = Pseudotrajectory("H2O", "H2O", rot_grid_origin=zg, trans_grid=trans_grid, rot_grid_body=zg)
    pt.generate_pseudotrajectory()
    file_name = f"{PATH_OUTPUT_PT}{pt.pt_name}.gro"
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
    zg = ZeroGrid()
    num_trans = 2
    trans_grid = TranslationParser("[1, 2]")
    distances = trans_grid.get_trans_grid()
    pt = Pseudotrajectory("H2O", "H2O", rot_grid_origin=ico_grid, trans_grid=trans_grid, rot_grid_body=zg)
    len_traj = pt.generate_pseudotrajectory()
    file_name = f"{PATH_OUTPUT_PT}{pt.pt_name}.gro"
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
        # TODO: assert orientation in internal basis is always the same?
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
    zg = ZeroGrid()
    num_trans = 3
    trans_grid = TranslationParser("[1, 2, 3]")
    distances = trans_grid.get_trans_grid()
    pt = Pseudotrajectory("H2O", "NH3", rot_grid_origin=zg, trans_grid=trans_grid, rot_grid_body=ico_grid)
    len_traj = pt.generate_pseudotrajectory()
    file_name = f"{PATH_OUTPUT_PT}{pt.pt_name}.gro"
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
    distances = trans_grid.get_trans_grid()
    pt = Pseudotrajectory("H2O", "NH3", rot_grid_origin=grid_o, trans_grid=trans_grid, rot_grid_body=grid_b)
    len_traj = pt.generate_pseudotrajectory()
    file_name = f"{PATH_OUTPUT_PT}{pt.pt_name}.gro"
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


if __name__ == '__main__':
    freshly_create_all_folders()
    copy_examples()
    test_pt_len()
    test_pt_translations()
    test_pt_rotations_origin()
    test_pt_rotations_body()
    test_order_of_operations()
