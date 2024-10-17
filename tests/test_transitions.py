import sys

import numpy as np
from numpy.typing import NDArray
from MDAnalysis import Universe


from molgri.io import OneMoleculeReader
from molgri.space.fullgrid import FullGrid
from molgri.molecules.transitions import AssignmentTool
from molgri.molecules.pts import Pseudotrajectory
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT, PATH_OUTPUT_AUTOSAVE
from molgri.plotting.create_vmdlog import show_assignments

def _create_grid(o="15", b="9", t="[0.2, 0.3, 0.4]"):
    my_array = FullGrid(b_grid_name=b, o_grid_name=o, t_grid_name=t)
    return my_array


def _create_pseudotraj(grid: NDArray, m1="input/H2O.gro", m2="input/H2O.gro"):
    om1 = OneMoleculeReader(m1).get_molecule()
    om2 = OneMoleculeReader(m2).get_molecule()
    pt = Pseudotrajectory(om1, om2, grid)
    return pt.get_pt_as_universe(), om2


def test_position_grid_assignments():
    # if I input a pt and same FullGrid, the first n_b assignments are to 0th position, then 1st ...
    my_grid = _create_grid()
    my_grid_array = my_grid.get_full_grid_as_array()
    my_pt_universe, my_second_mol = _create_pseudotraj(my_grid_array)

    at = AssignmentTool(my_grid.get_full_grid_as_array(), my_pt_universe, my_second_mol)
    n_b = my_grid.get_b_N()
    n_position_grid = my_grid.get_o_N() * my_grid.get_t_N()
    repeated_natural_num = np.repeat(np.arange(n_position_grid), n_b)
    assert np.all(repeated_natural_num == at._get_position_assignments())
    # with the same pt but a smaller fullgrid, I expect an equal number of structures in every cell
    num_o = 500
    assignment_grid = _create_grid(b="8", o="12", t="[0.2, 0.3, 0.4]")
    assignment_grid_array = assignment_grid.get_full_grid_as_array()
    pt_grid = _create_grid(b="8", o=str(num_o), t="[0.2, 0.3, 0.4]")
    pt_grid_array = pt_grid.get_full_grid_as_array()
    my_pt_universe, my_second_mol = _create_pseudotraj(pt_grid_array)
    at = AssignmentTool(assignment_grid_array, my_pt_universe, my_second_mol)
    nums, counts = np.unique(at._get_position_assignments(), return_counts=True)
    assert np.all(nums == np.arange(12*3))
    # counts approx the same
    expected_per_position = 8*num_o/12
    # there will be errors from expected distribution, partially cause the areas are not of exactly the same size,
    # but especially because there is a bunch of pt points exactly in the middle which we don't expect
    rel_errors = np.abs(counts-expected_per_position)/expected_per_position * 100
    print(rel_errors)
    assert np.all(rel_errors < 15)


def test_quaternion_grid_assignments():
    # if I input a pt and same FullGrid, assignments should be 0, 1, ... n_b, 0, 1... n_b, 0 ......
    my_grid = _create_grid(b="17", o="12", t="[0.2, 0.3, 0.4]")
    my_pt_universe, my_second_mol = _create_pseudotraj(my_grid.get_full_grid_as_array())
    at = AssignmentTool(my_grid.get_full_grid_as_array(), my_pt_universe, my_second_mol)
    n_b = my_grid.get_b_N()
    n_position_grid = my_grid.get_o_N() * my_grid.get_t_N()
    repeated_natural_num = np.tile(np.arange(n_b), n_position_grid)
    assert np.all(repeated_natural_num == at._get_quaternion_assignments())

    """
    Here we create simulation histogram for 40 orientation, then assign it to a full grid with 8 orientation. We 
    expect that we will still get a uniform distribution among the 8 orientation classes.
    """
    num_b = 500
    num_assignment_b = 8
    assignment_grid = _create_grid(b=f"{num_assignment_b}", o="20", t="[0.2, 0.3, 0.4, 0.5]")
    assignment_grid_array = assignment_grid.get_full_grid_as_array()
    pt_grid = _create_grid(b=f"randomQ_{num_b}", o="20", t="[0.2, 0.3, 0.4, 0.5]")
    pt_grid_array = pt_grid.get_full_grid_as_array()
    my_pt_universe, my_second_mol = _create_pseudotraj(pt_grid_array)
    at = AssignmentTool(assignment_grid_array, my_pt_universe, my_second_mol)
    np.set_printoptions(threshold=sys.maxsize)
    nums, counts = np.unique(at._get_quaternion_assignments(), return_counts=True)
    assert np.all(nums == np.arange(num_assignment_b))
    # counts approx the same
    expected_per_position = 20 * 4 * num_b / num_assignment_b
    # there will be errors from expected distribution, partially cause the areas are not of exactly the same size,
    # but especially because there is a bunch of pt points exactly in the middle which we don't expect
    rel_errors = np.abs(counts - expected_per_position) / expected_per_position * 100
    print(counts, expected_per_position, rel_errors)
    assert np.all(rel_errors < 15)


def test_full_grid_assignments():

    # perfect divisions, same sizes
    my_grid = _create_grid(b="8", o="12", t="[0.2, 0.3, 0.4]")
    my_pt_universe, my_second_mol = _create_pseudotraj(my_grid.get_full_grid_as_array())
    at = AssignmentTool(my_grid.get_full_grid_as_array(), my_pt_universe, my_second_mol)
    print(at.get_full_assignments())
    assert np.all(np.arange(len(at.trajectory_universe.trajectory)) == at.get_full_assignments())

    # non-perfect divisions, same sizes
    my_grid = _create_grid(b="17", o="25", t="[0.2, 0.3, 0.4]")
    my_pt_universe, my_second_mol = _create_pseudotraj(my_grid.get_full_grid_as_array())
    at = AssignmentTool(my_grid.get_full_grid_as_array(), my_pt_universe, my_second_mol)
    assert np.all(np.arange(len(at.trajectory_universe.trajectory)) == at.get_full_assignments())


if __name__ == "__main__":
    #test_position_grid_assignments()
    #view_quaternion_assignments()
    test_quaternion_grid_assignments()
    test_full_grid_assignments()

