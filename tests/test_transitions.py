import sys

import numpy as np
from molgri.space.fullgrid import FullGrid

from molgri.molecules.transitions import SimulationHistogram
from molgri.molecules.writers import PtIOManager

def _create_sim_hist(m1="H2O", m2="H2O", o="15", b="9", t="[0.2, 0.3, 0.4]", second_molecule_selection="bynum 4:6",
                     full_grid=None):
    my_pt_generator = PtIOManager(m1, m2, o_grid_name=o, b_grid_name=b, t_grid_name=t)
    my_pt_generator.construct_pt()
    my_pt_name = my_pt_generator.get_name()

    my_sh = SimulationHistogram(my_pt_name, is_pt=True, second_molecule_selection=second_molecule_selection,
                                full_grid=full_grid, use_saved = False)
    return my_sh


def test_position_grid_assignments():
    # if I input a pt and same FullGrid, the first n_b assignments are to 0th position, then 1st ...
    sh_same_fg = _create_sim_hist()
    n_b = sh_same_fg.full_grid.b_rotations.get_N()
    n_position_grid = len(sh_same_fg.full_grid.get_position_grid_as_array())
    repeated_natural_num = np.repeat(np.arange(n_position_grid), n_b)
    assert np.all(repeated_natural_num==sh_same_fg.get_position_assignments())
    # with the same pt but a smaller fullgrid, I expect an equal number of structures in every cell
    num_o = 500
    sh_diff_fg = _create_sim_hist(o=f"{num_o}", b="8",
                                  full_grid=FullGrid(b_grid_name="8", o_grid_name="12", t_grid_name="[0.2, 0.3, "
                                                                                                      "0.4]"))
    nums, counts = np.unique(sh_diff_fg.get_position_assignments(), return_counts=True)
    assert np.all(nums == np.arange(12*3))
    # counts approx the same
    expected_per_position = 8*num_o/12
    # there will be errors from expected distribution, partially cause the areas are not of exactly the same size,
    # but especially because there is a bunch of pt points exactly in the middle which we don't expect
    rel_errors = np.abs(counts-expected_per_position)/expected_per_position * 100
    print(np.max(rel_errors))
    assert np.all(rel_errors < 20)


def test_quaternion_grid_assignments():
    # if I input a pt and same FullGrid, assignments should be 0, 1, ... n_b, 0, 1... n_b, 0 ......
    sh_same_fg = _create_sim_hist(b="17", o="12",
                                  full_grid=FullGrid(b_grid_name="17", o_grid_name="12", t_grid_name="[0.2, 0.3, 0.4]"))
    ideal_q_indices = sh_same_fg.full_grid.get_quaternion_index()
    assert np.all(ideal_q_indices == sh_same_fg.get_quaternion_assignments())

    # what happens if not exact match
    num_b = 40
    sh_same_fg = _create_sim_hist(b=f"{num_b}", o="20", t="[0.2, 0.3, 0.4, 0.5]",
                                  full_grid=FullGrid(b_grid_name="8", o_grid_name="12", t_grid_name="[0.2, 0.3, 0.4]"))
    np.set_printoptions(threshold=sys.maxsize)
    assignments = sh_same_fg.get_quaternion_assignments()
    nums, counts = np.unique(sh_same_fg.get_quaternion_assignments(), return_counts=True)
    assert np.all(nums == np.arange(8))
    # counts approx the same
    expected_per_position = 20 * 4 * num_b / 8
    # there will be errors from expected distribution, partially cause the areas are not of exactly the same size,
    # but especially because there is a bunch of pt points exactly in the middle which we don't expect
    rel_errors = np.abs(counts - expected_per_position) / expected_per_position * 100
    print(np.max(rel_errors))
    assert np.all(rel_errors < 10)


def test_full_grid_assignments():

    # perfect divisions, same sizes
    sh_same_fg = _create_sim_hist(b="8", o="12",
                                  full_grid=FullGrid(b_grid_name="8", o_grid_name="12", t_grid_name="[0.2, 0.3, 0.4]"))
    len_traj = len(sh_same_fg)
    assert np.all(sh_same_fg.get_full_assignments() == np.arange(len_traj))

    # non-perfect divisions, same sizes
    sh_same_fg = _create_sim_hist(b="17", o="25",
                                  full_grid=FullGrid(b_grid_name="17", o_grid_name="25", t_grid_name="[0.2, 0.3, 0.4]"))
    len_traj = len(sh_same_fg)
    assert np.all(sh_same_fg.get_full_assignments() == np.arange(len_traj))

    # different sizes of b grid
    # TODO: this one is not that great
    # sh_same_fg = _create_sim_hist(b="17", o="15",
    #                               full_grid=FullGrid(b_grid_name="10", o_grid_name="15", t_grid_name="[0.2, 0.3, 0.4]"))
    # assignments = sh_same_fg.get_full_assignments()
    # print(np.unique(assignments, return_counts=True))
    #print(np.where(assignments!=np.arange(len(sh_same_fg.full_grid.get_full_grid_as_array()))))

    # different sizes of o grid
    sh_same_fg = _create_sim_hist(b="8", o="40",
                                  full_grid=FullGrid(b_grid_name="8", o_grid_name="10", t_grid_name="[0.2, 0.3, 0.4]"))
    assignments = sh_same_fg.get_full_assignments()
    print(np.unique(assignments, return_counts=True))


def test_pindex_and_quindex():
    # cells that are considered neighbours should have same pindex, nex
    pass

if __name__ == "__main__":
    test_position_grid_assignments()
    test_quaternion_grid_assignments()
    test_full_grid_assignments()

