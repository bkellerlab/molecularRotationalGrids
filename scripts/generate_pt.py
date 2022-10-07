"""
This is a user-friendly script for generating a pseudotrajectory.
"""

import argparse
from os.path import exists
from ast import literal_eval

from molgri.grids.grid import build_grid
from molgri.my_constants import ENDING_GRID_FILES
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT, PATH_OUTPUT_ROTGRIDS
from molgri.parsers.name_parser import NameParser

# TODO: define total_N and generate in all dimensions uniform grid?
# TODO: allow different rotation grids for two types of rotation

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
    rot_grid = build_grid(algo, N, use_saved=True, print_message=True)
    # assert translational grid is a tuple with three parameters
    trans_grid = literal_eval(args.transgrid)
    assert len(trans_grid) == 3, "Translational grid must be provided as a tuple of three parameters."
    for item in trans_grid:
        assert isinstance(item, int), "Items in translation grid tuple should be integers."
    return rot_grid


if __name__ == '__main__':
    my_args = parser.parse_args()
    check_file_existence(my_args)