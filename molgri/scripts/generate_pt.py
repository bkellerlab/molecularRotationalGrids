"""
This is a user-friendly script for generating a pseudotrajectory.
"""

import argparse
from os.path import exists

from ..paths import PATH_INPUT_BASEGRO
from molgri.parsers import NameParser, TranslationParser
from ..scripts.generate_grid import prepare_grid
from ..scripts.set_up_io import freshly_create_all_folders

# TODO: define total_N and generate in all dimensions uniform grid?
# TODO: allow different rotation grids for two types of rotation
from molgri.pts import Pseudotrajectory

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-m1', type=str, nargs='?', required=True,
                    help='name of the .gro file containing the fixed molecule')
requiredNamed.add_argument('-m2', type=str, nargs='?', required=True,
                    help='name of the .gro file containing the mobile molecule')
requiredNamed.add_argument('-rotgrid', metavar='rg', type=str, nargs='?', required=True,
                    help='name of the rotation grid in the form algorithm_N (eg. ico_50)')
requiredNamed.add_argument('-transgrid', metavar='tg', type=str, nargs='?', required=True,
                    help='translation grid provided as a list of distances, as linspace(start, stop, num) '
                         'or range(start, stop, step) in nanometers')
parser.add_argument('--only_origin', action='store_true',
                    help='only include rotations around the origin, not around the body')
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')


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
    # parse translational grid
    trans_grid = TranslationParser(args.transgrid)
    return rot_grid, trans_grid


def prepare_pseudotrajectory(args, r_grid, t_grid):
    if args.only_origin:
        traj_type = "circular"
    else:
        traj_type = "full"
    pt = Pseudotrajectory(args.m1, args.m2, rot_grid=r_grid, trans_grid=t_grid, traj_type=traj_type)
    end_index = pt.generate_pt_and_time()
    print(f"Generated a {pt.decorator_label} with {end_index} timesteps.")


def run_generate_pt():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    my_rg, my_tg = check_file_existence(my_args)
    prepare_pseudotrajectory(my_args, my_rg, my_tg)


if __name__ == '__main__':
    run_generate_pt()
