import numpy as np

from molgri.pts import Pseudotrajectory
from molgri.grids import IcoGrid, Cube4DGrid, ZeroGrid
from molgri.parsers import TranslationParser, MultiframeGroParser
from molgri.scripts.set_up_io import freshly_create_all_folders, copy_examples
from molgri.paths import PATH_OUTPUT_PT
from molgri.utils import angle_between_vectors


def test_pt_len():
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
    # on a zero rotation grids
    zg = ZeroGrid()
    trans_grid = TranslationParser("range(1, 5, 0.5)")
    distances = trans_grid.get_trans_grid()
    pt = Pseudotrajectory("H2O", "H2O", rot_grid_origin=zg, trans_grid=trans_grid, rot_grid_body=zg)
    pt.generate_pseudotrajectory()
    file_name = f"{PATH_OUTPUT_PT}{pt.pt_name}.gro"
    # center of mass of the second molecule moves in z direction
    ts = MultiframeGroParser(file_name).timesteps
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
    ts = MultiframeGroParser(file_name, parse_atoms=True).timesteps
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
    ts = MultiframeGroParser(file_name, parse_atoms=True).timesteps
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
        basis = ts[frame_i].molecule_set.all_objects[1].basis
        vec_atom1 = ts[frame_i].molecule_set.all_objects[1].atoms[0].position
        print(basis, vec_atom1)


if __name__ == '__main__':
    freshly_create_all_folders()
    copy_examples()
    test_pt_len()
    test_pt_translations()
    test_pt_rotations_origin()
    test_pt_rotations_body()
