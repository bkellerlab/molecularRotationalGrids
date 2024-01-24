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
    sh_diff_fg = _create_sim_hist(o="162",
                                  full_grid=FullGrid(b_grid_name="9", o_grid_name="12", t_grid_name="[0.2, 0.3, "
                                                                                                      "0.4]"))
    nums, counts = np.unique(sh_diff_fg.get_position_assignments(), return_counts=True)
    assert np.all(nums == np.arange(12*3))
    # counts approx the same
    expected_per_position = 9*162/12
    rel_errors = np.abs(counts-expected_per_position)/expected_per_position * 100
    assert np.all(rel_errors < 14)

def test_quaternion_grid_assignments():
    # if I input a pt and same FullGrid, assignments should be 0, 1, ... n_b, 0, 1... n_b, 0 ......
    sh_same_fg = _create_sim_hist(b="8", o="15",
                                  full_grid=FullGrid(b_grid_name="8", o_grid_name="15", t_grid_name="[0.2, 0.3, 0.4]"))
    n_b = sh_same_fg.full_grid.b_rotations.get_N()
    n_position_grid = len(sh_same_fg.full_grid.get_position_grid_as_array())
    repeated_natural_num = np.tile(np.arange(n_position_grid), n_b)
    print(sh_same_fg.get_quaternion_assignments())
    #44, 39, 41, 50, 52, 45, 41, 48
    print(np.unique(sh_same_fg.get_quaternion_assignments(), return_counts=True))
    #assert np.all(repeated_natural_num==sh_same_fg.get_quaternion_assignments())




if __name__ == "__main__":
    test_position_grid_assignments()
    test_quaternion_grid_assignments()