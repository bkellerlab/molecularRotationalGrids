"""
This is a user-friendly script for generating a pseudotrajectory.
"""

import argparse

from molgri.constants import EXTENSION_TOPOLOGY, EXTENSION_TRAJECTORY
from molgri.writers import PtIOManager
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
parser.add_argument('--as_dir', action='store_true',
                    help='Save the PT as a directory of frames')
parser.add_argument('--extension_trajectory', type=str, default=EXTENSION_TRAJECTORY,
                    help=f"File extension for generated (pseudo)-trajectories [default: {EXTENSION_TRAJECTORY}]")
parser.add_argument('--extension_topology', type=str, default=EXTENSION_TOPOLOGY,
                    help=f"File extension for generated topologies [default: {EXTENSION_TOPOLOGY}]")


def run_generate_pt():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    manager = PtIOManager(name_central_molecule=my_args.m1, name_rotating_molecule=my_args.m2,
                          b_grid_name=my_args.bodygrid, o_grid_name=my_args.origingrid, t_grid_name=my_args.transgrid)
    manager.construct_pt_and_time(as_dir=my_args.as_dir, extension_trajectory=my_args.extension_trajectory,
                                  extension_structure=my_args.extension_topology)


if __name__ == '__main__':
    run_generate_pt()
