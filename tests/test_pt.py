from molgri.pts import Pseudotrajectory
from molgri.grids import IcoGrid, Cube4DGrid, ZeroGrid
from molgri.parsers import TranslationParser
from molgri.scripts.set_up_io import freshly_create_all_folders, copy_examples


def test_pt_len():
    # origin rot grid = body rot grid
    num_rotations = 55
    rot_grid = IcoGrid(num_rotations)
    trans_grid = TranslationParser("linspace(1, 5, 10)")
    num_translations = trans_grid.get_N_trans()
    pt = Pseudotrajectory("H2O", "NH3", rot_grid_origin=rot_grid, trans_grid=trans_grid, rot_grid_body=rot_grid)
    end_index = pt.generate_pseudotrajectory()
    assert end_index == num_translations*num_rotations*num_rotations
    # origin rot grid =/= body rot grid
    num_body = 3
    body_grid = IcoGrid(num_body)
    num_origin = 18
    origin_grid = Cube4DGrid(num_origin)
    trans_grid = TranslationParser("range(1, 5, 0.5)")
    num_translations = trans_grid.get_N_trans()
    pt = Pseudotrajectory("H2O", "NH3", rot_grid_origin=origin_grid, trans_grid=trans_grid, rot_grid_body=body_grid)
    end_index = pt.generate_pseudotrajectory()
    assert end_index == num_translations * num_body * num_origin
    # body grid is null (only_origin)
    num_origin = 9
    origin_grid = Cube4DGrid(num_origin)
    trans_grid = TranslationParser("range(1, 5, 0.5)")
    num_translations = trans_grid.get_N_trans()
    pt = Pseudotrajectory("H2O", "NH3", rot_grid_origin=origin_grid, trans_grid=trans_grid, rot_grid_body=None)
    end_index = pt.generate_pseudotrajectory()
    assert end_index == num_translations * num_origin
    # TODO: assert that this also equals numbers in the file


def test_pt_translations():
    # on a zero rotation grids
    zg = ZeroGrid()
    trans_grid = TranslationParser("range(1, 5, 0.5)")
    distances = trans_grid.get_trans_grid()
    pt = Pseudotrajectory("H2O", "H2O", rot_grid_origin=zg, trans_grid=trans_grid, rot_grid_body=zg)
    # TODO: assert distances written in a file


if __name__ == '__main__':
    freshly_create_all_folders()
    copy_examples()
    test_pt_len()
    test_pt_translations()
