import numpy as np

from molgri.network.polytope import Cube4DPolytope, IcosahedronPolytope
from molgri.utils.arrays import all_rows_unique
from molgri.utils.quaternions import random_quaternions
from molgri.molecules.find_unit_cell import get_x_y_grid_inputs

PATH_INPUT_MOLECULES = "inputs/one_molecule_structures/"


def auto_create_grid(config_data: dict):
    """
    In case you want to automatically construct the grid in x and y direction based on crystal cell.
    """
    if config_data["grid"]["translation_algorithm"] != "cartesian":
        return config_data
    all_translation_coo = np.array(config_data["grid"]["translation_subgrids_A"]).reshape(-1)
    if 'None' in all_translation_coo:
        path_input = f"{PATH_INPUT_MOLECULES}{config_data['pseudotrajectory']['molecule_1']}.gro"
        num_x_points = config_data["grid"]["translation_subgrids_A"][0][2]
        num_y_points = config_data["grid"]["translation_subgrids_A"][1][2]
        z = config_data["grid"]["translation_subgrids_A"][2]
        x, y = get_x_y_grid_inputs(path_input, num_x_points=num_x_points, num_y_points=num_y_points)
        config_data["grid"]["translation_subgrids_A"] = [x, y, z]
    config_data["grid"]["GRID_LENGTH"] = config_data["grid"]["N_rotations"]
    return config_data



def _make_cartesian_grid(config_data: dict, save_information: dict):
    subgrids_A = config_data["grid"]["translation_subgrids_A"]
    periodic_in = config_data["grid"]["periodic_in"]
    all_grid_limits_A = []
    all_grid_N_points = []
    for subgrid in subgrids_A:
        all_grid_limits_A.append(subgrid[:2])
        all_grid_N_points.append(subgrid[2])
    save_information["subgrid_limits_A"] = all_grid_limits_A
    save_information["subgrid_N_points"] = all_grid_N_points
    save_information["periodic_in"] = periodic_in

    grid_labels = ["x", "y", "z"]
    grid_deltas = []
    sub_grids = []
    for grid_label, grid_linspace, is_peridic in zip(grid_labels, subgrids_A, periodic_in):
        start_point, end_point, num_points = grid_linspace
        # if it is periodic you should not be using the end point for the grid as it would be identical to start point
        sub_grid = np.linspace(start_point, end_point, num_points, endpoint=(not is_peridic))
        # we start at largest distances and go to the smallest so that the 1st energy isn't too high (GROMACS complains)
        if grid_label == "z" and start_point < end_point:
            sub_grid = sub_grid[::-1]
        if len(sub_grid) >= 2:
            grid_deltas.append(np.abs(sub_grid[1] - sub_grid[0]))
        else:
            grid_deltas.append(0.0)
        sub_grids.append(sub_grid)
    # for yaml it is important to save to list
    save_information["translation_grid_deltas"] = grid_deltas
    save_information["explain_subgrid_points"] = ["x_grid_in_angstrom", "y_grid_in_angstrom", "z_grid_in_angstrom"]
    save_information["subgrid_points"] = sub_grids
    return save_information


def _make_spherical_grid(config_data: dict, save_information: dict):
    subgrids_A = config_data["grid"]["translation_subgrids_A"]
    random_seed = config_data["grid"]["rotation_random_seed"]

    # first subgrid is on unit sphere
    spherical_N_points = subgrids_A[0]
    ico = IcosahedronPolytope()
    ico.create_exactly_N_points(spherical_N_points, random_seed)
    spherical_points = ico.get_nodes(projection=True)
    # the second one is radial
    num_start, num_stop, num_steps = subgrids_A[1]
    r_grid = np.linspace(num_start, num_stop, num_steps)
    if num_start < num_stop:
        r_grid = r_grid[::-1]

    # periodic makes no sense here
    periodic_in = config_data["grid"]["periodic_in"]
    assert np.all(~np.array(periodic_in)), ("You are trying to use periodicity with spherical grid, how should that "
                                            "work?")

    # now save all info
    save_information["random_seed"] = random_seed
    save_information["periodic_in"] = periodic_in
    save_information["translation_grid_deltas"] = [None, np.abs(r_grid[1]-r_grid[0])]
    save_information["explain_subgrid_points"] = ["points_on_unit_sphere", "radial_grid_in_angstrom"]
    save_information["subgrid_points"] = [spherical_points, r_grid]
    save_information["subgrid_N_points"] = spherical_N_points*num_steps
    save_information["subgrid_limits_A"] = [[None, None], [None, None],[None, None]]
    return save_information


def _make_translation_grid(config_data: dict, save_information: dict):
    translation_algorithm = config_data["grid"]["translation_algorithm"]
    save_information["translation_algorithm"] = translation_algorithm
    if translation_algorithm == "cartesian":
        return _make_cartesian_grid(config_data, save_information)
    elif translation_algorithm == "spherical":
        return _make_spherical_grid(config_data, save_information)
    else:
        raise KeyError(f"{translation_algorithm} is not a valid translation algorithm keyword")


def _make_rotation_grid(config_data: dict, save_information: dict):
    rotation_algorithm = config_data["grid"]["rotation_algorithm"]
    N_rotations = config_data["grid"]["N_rotations"]
    rotation_random_seed = config_data["grid"]["rotation_random_seed"]
    # people may interpret "no rotations" as zero, but actually that means we are using exactly one rotation quaternion
    if N_rotations == 0:
        N_rotations = 1

    save_information["rotation_algorithm"] = rotation_algorithm
    save_information["N_rotations"] = N_rotations
    save_information["random_seed"] = rotation_random_seed

    # for exactly one rotation just use identity
    if N_rotations == 1:
        quaternions = np.array([[1, 0, 0, 0]])
    elif rotation_algorithm == "random":
        quaternions = random_quaternions(N_rotations, only_upper=True,
                                         rotation_random_seed=rotation_random_seed)
    elif rotation_algorithm == "hypercube":
        polytope = Cube4DPolytope()
        quaternions = polytope.create_exactly_N_points(N_rotations, rotation_random_seed=rotation_random_seed)
    else:
        raise KeyError(f"{N_rotations} is not a valid rotation algorithm keyword")
    # test that we have the right number of unique quaternions
    assert len(quaternions) == N_rotations
    all_rows_unique(quaternions)
    save_information["quaternions"] = quaternions.tolist()
    return save_information


def make_grid_base(config_data: dict):
    """
    This will create all the grid bases (eg x, y, z and quaternion or r, spherical and quaternion) without creating a
    complex network yet.

    Args:
        config_data (dict): data given by the config_data file
    Returns:
        a dictionary with necessary information to build a network
    """
    save_information = _make_translation_grid(config_data, dict())
    save_information = _make_rotation_grid(config_data, save_information)
    save_information["N_translations"] = int(np.prod(save_information["subgrid_N_points"]))
    save_information["N_total"] = save_information["N_translations"] * save_information["N_rotations"]
    return save_information


