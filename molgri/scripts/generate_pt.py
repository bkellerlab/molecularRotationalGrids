"""
This is a user-friendly script for generating a pseudotrajectory.
"""

import argparse
from os.path import exists
from ast import literal_eval

import numpy as np

from ..paths import PATH_INPUT_BASEGRO
from molgri.parsers import NameParser
from ..scripts.generate_grid import prepare_grid
from ..scripts.set_up_io_directories import freshly_create_all_folders

# TODO: define total_N and generate in all dimensions uniform grid?
# TODO: allow different rotation grids for two types of rotation
from molgri.pts import Pseudotrajectory

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-m1', type=str, nargs='?', required=True,
                    help='Which molecule should be fixed at the center (eg. H2O)?')
requiredNamed.add_argument('-m2', type=str, nargs='?', required=True,
                    help='Which molecule should be rotating around the first (eg. H2O)?')
requiredNamed.add_argument('-rotgrid', metavar='rg', type=str, nargs='?', required=True,
                    help='Name of the rotation grid file.')
requiredNamed.add_argument('-transgrid', metavar='tg', type=str, nargs='?', required=True,
                    help='Radii of rotation of one molecule around the other [nm].')
parser.add_argument('--only_origin', action='store_true',
                    help='Only include rotations around the origin.')
parser.add_argument('--recalculate', action='store_true',
                    help='Even if a saved version of this grid already exists, recalculate it.')


def check_file_existence(args):
    path_c_mol = f"{PATH_INPUT_BASEGRO}{args.m1}.gro"
    path_r_mol = f"{PATH_INPUT_BASEGRO}{args.m2}.gro"
    # check if rotgrid name a valid name
    nap = NameParser(args.rotgrid)
    N = nap.num_grid_points
    algo = nap.grid_type
    assert algo is not None, f"Rotation grid algorithm not recognised, check rotgrid argument: {args.rotgrid}"
    assert N is not None, f"Num of grid points not recognised, check rotgrid argument: {args.rotgrid}"
    # check if input rotgrid files exist
    if not exists(path_c_mol):
        raise FileNotFoundError(f"Could not find the file {args.m1}.gro at {PATH_INPUT_BASEGRO}. "
                                "Please provide a valid .gro file name as the first script parameter.")
    if not exists(path_r_mol):
        raise FileNotFoundError(f"Could not find the file {args.m2}.gro at {PATH_INPUT_BASEGRO}. "
                                "Please provide a valid .gro file name as the second script parameter.")
    # if the rotational grid file doesn't exist, create it
    rot_grid = prepare_grid(args, nap)
    # assert translational grid is a tuple with three parameters
    trans_grid = literal_eval(args.transgrid)
    assert len(trans_grid) == 3, "Translational grid must be provided as a tuple of three parameters."
    for item in trans_grid:
        assert isinstance(item, int), "Items in translation grid tuple should be integers."
    # TODO: linspace is in general NOT a good enough approximation
    trans_grid = np.linspace(*trans_grid)
    return rot_grid, trans_grid


def prepare_pseudotrajectory(args, r_grid, t_grid):
    if args.only_origin:
        traj_type = "circular"
    else:
        traj_type = "full"
    pt = Pseudotrajectory(args.m1, args.m2, grid=r_grid, traj_type=traj_type)
    end_index = pt.generate_pt_and_time(initial_distance_nm=t_grid[0], radii=t_grid)
    print(f"Generated a {pt.decorator_label} with {end_index} timesteps.")


def run_generate_pt():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    my_rg, my_tg = check_file_existence(my_args)
    prepare_pseudotrajectory(my_args, my_rg, my_tg)


if __name__ == '__main__':
    run_generate_pt()
