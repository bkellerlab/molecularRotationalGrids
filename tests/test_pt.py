import numpy as np

from molgri.pts import Pseudotrajectory
from molgri.grids import IcoGrid, Cube4DGrid, ZeroGrid
from molgri.parsers import TranslationParser, MultiframeGroParser
from molgri.scripts.set_up_io import freshly_create_all_folders, copy_examples
from molgri.paths import PATH_OUTPUT_PT


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
    # TODO: assert distances written in a file
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


if __name__ == '__main__':
    freshly_create_all_folders()
    copy_examples()
    test_pt_len()
    test_pt_translations()
