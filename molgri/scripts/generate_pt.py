"""
This is a user-friendly script for generating a pseudotrajectory.
"""

import argparse

from molgri.constants import EXTENSION_TOPOLOGY, EXTENSION_TRAJECTORY
from molgri.molecules.writers import PtIOManager
from ..scripts.set_up_io import freshly_create_all_folders

# TODO: define total_N and generate in all dimensions uniform grid?

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-m1', type=str, nargs='?', required=True,
                           help='name of the .gro file containing the fixed molecule')
requiredNamed.add_argument('-m2', type=str, nargs='?', required=True,
                           help='name of the .gro file containing the mobile molecule')
requiredNamed.add_argument('-origingrid', metavar='og', type=str, nargs='?', required=True,
                           help='name of the rotation grid for rotations around origin in the form '
                                'algorithm_N (eg. ico_50) OR just a number (eg. 50)')
requiredNamed.add_argument('-bodygrid', metavar='bg', type=str, nargs='?', required=True,
                           help='name of the rotation grid for rotations around body in the form '
                                'algorithm_N (eg. ico_50) OR just a number (eg. 50) '
                                'OR None if you only want rotations about origin')
requiredNamed.add_argument('-transgrid', metavar='tg', type=str, nargs='?', required=True,
                           help='translation grid provided as a list of distances, as linspace(start, stop, num) '
                                'or range(start, stop, step) in nanometers')
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')
parser.add_argument('--only_origin', action='store_true',
                    help='An outdated function to suppress body rotations. We suggest using -b zero instead')
parser.add_argument('--as_dir', action='store_true',
                    help='Save the PT as a directory of frames')
parser.add_argument('--extension_trajectory', type=str, default=EXTENSION_TRAJECTORY,
                    help=f"File extension for generated (pseudo)-trajectories [default: {EXTENSION_TRAJECTORY}]")
parser.add_argument('--extension_structure', type=str, default=EXTENSION_TOPOLOGY,
                    help=f"File extension for generated topologies [default: {EXTENSION_TOPOLOGY}]")


def run_generate_pt():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    if my_args.only_origin:
        my_args.bodygrid = "zero"
        print("Warning: the flag --only_origin is deprecated. We suggest setting -b zero instead.")
    if "ico" in my_args.bodygrid or "cube3D" in my_args.bodygrid:
        print(f"Warning! You are using -b {my_args.bodygrid} to create the grid of rotations around the COM. "
              f"We do not recommend the use of Icosahedron or 3D cube grids for this purpose; they are optimised "
              f"for generation of origin rotations (-o). We suggest using a 4D cube grid (cube4D) instead.")
    manager = PtIOManager(name_central_molecule=my_args.m1, name_rotating_molecule=my_args.m2,
                          b_grid_name=my_args.bodygrid, o_grid_name=my_args.origingrid, t_grid_name=my_args.transgrid)
    manager.construct_pt_and_time(as_dir=my_args.as_dir, extension_trajectory=my_args.extension_trajectory,
                                  extension_structure=my_args.extension_structure)


if __name__ == '__main__':
    run_generate_pt()

