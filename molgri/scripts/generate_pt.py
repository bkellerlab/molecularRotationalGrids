"""
This is a user-friendly script for generating a pseudotrajectory.
"""

import argparse
from os.path import exists

from ..paths import PATH_INPUT_BASEGRO
from molgri.parsers import NameParser, TranslationParser
from ..scripts.generate_grid import prepare_grid
from molgri.grids import FullGrid
from molgri.writers import PtWriter
from ..scripts.set_up_io import freshly_create_all_folders

# TODO: define total_N and generate in all dimensions uniform grid?
from molgri.pts import Pseudotrajectory

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-m1', type=str, nargs='?', required=True,
                    help='name of the .gro file containing the fixed molecule')
requiredNamed.add_argument('-m2', type=str, nargs='?', required=True,
                    help='name of the .gro file containing the mobile molecule')
requiredNamed.add_argument('-origingrid', metavar='og', type=str, nargs='?', required=True,
                    help='name of the rotation grid for rotations around origin in the form algorithm_N (eg. ico_50)')
requiredNamed.add_argument('-bodygrid', metavar='bg', type=str, nargs='?', required=True,
                    help='name of the rotation grid for rotations around body in the form algorithm_N (eg. ico_50)'
                         'OR None if you only want rotations arount origin')
requiredNamed.add_argument('-transgrid', metavar='tg', type=str, nargs='?', required=True,
                    help='translation grid provided as a list of distances, as linspace(start, stop, num) '
                         'or range(start, stop, step) in nanometers')
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')


def check_file_existence(args):
    path_c_mol = f"{PATH_INPUT_BASEGRO}{args.m1}.gro"
    path_r_mol = f"{PATH_INPUT_BASEGRO}{args.m2}.gro"
    # check rotational grid names
    nap1 = NameParser(args.origingrid)
    N = nap1.num_grid_points
    algo = nap1.grid_type
    assert algo is not None, f"Rotation grid algorithm not recognised, check origingrid argument: {args.origingrid}"
    assert N is not None, f"Num of grid points not recognised, check origingrid argument: {args.origingrid}"
    if args.bodygrid in ["None", "none", "NONE"]:
        nap2 = NameParser("zero_1")
    else:
        nap2 = NameParser(args.bodygrid)
    N = nap2.num_grid_points
    algo = nap2.grid_type
    assert algo is not None, f"Rotation grid algorithm not recognised, check bodygrid argument: {args.bodygrid}"
    assert N is not None, f"Num of grid points not recognised, check bodygrid argument: {args.bodygrid}"
    # check if input rotgrid files exist
    if not exists(path_c_mol):
        raise FileNotFoundError(f"Could not find the file {args.m1}.gro at {PATH_INPUT_BASEGRO}. "
                                "Please provide a valid .gro file name as the first script parameter.")
    if not exists(path_r_mol):
        raise FileNotFoundError(f"Could not find the file {args.m2}.gro at {PATH_INPUT_BASEGRO}. "
                                "Please provide a valid .gro file name as the second script parameter.")
    # if the rotational grid file doesn't exist, create it
    origin_grid = prepare_grid(args, nap1)
    body_grid = prepare_grid(args, nap2)
    # parse translational grid
    trans_grid = TranslationParser(args.transgrid)
    return origin_grid, body_grid, trans_grid


def run_generate_pt():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    my_og, my_bg, my_tg = check_file_existence(my_args)
    full_grid = FullGrid(b_grid=my_bg, o_grid=my_og, t_grid=my_tg)
    pt_writer = PtWriter(my_args.m1, my_args.m2, full_grid)
    pt_writer.write_full_pt_gro()
    print(f"Generated a Pseudotrajectory with {end_index} timesteps.")


if __name__ == '__main__':
    run_generate_pt()
