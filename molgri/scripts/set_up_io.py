"""
This script should be run to define input and output directories if the user wants to select directories other
than default ones. It can also be run at any later point to move the contents from previously selected directories
to new ones.
"""
import argparse
import os
import shutil

from ..constants import PATH_USER_PATHS, PATH_EXAMPLES

parser = argparse.ArgumentParser()
parser.add_argument('--examples', action='store_true',
                    help='copy example inputs')

# if you introduce new IO folders, add them to next three lines, keep the order the same!
IO_FOLDER_PURPUSES = ["rotational grids", "pseudotrajectory files", "base gro files", "plots", "animations",
                      "grid statistics", "translational grids", "full grids", "cells", "energies"]
IO_FOLDER_DEFAULTS = ["output/rot_grids/", "output/pt_files/", "input/", "output/figures/",
                      "output/animations/", "output/statistics_files/", "output/trans_grids/", "output/full_grids/",
                      "output/cells/", "input/"]
IO_VARIABLE_NAMES = ["PATH_OUTPUT_ROTGRIDS", "PATH_OUTPUT_PT", "PATH_INPUT_BASEGRO", "PATH_OUTPUT_PLOTS",
                     "PATH_OUTPUT_ANIS", "PATH_OUTPUT_STAT", "PATH_OUTPUT_TRANSGRIDS", "PATH_OUTPUT_FULL_GRIDS",
                     "PATH_OUTPUT_CELLS", "PATH_INPUT_ENERGIES"]

assert len(IO_VARIABLE_NAMES) == len(IO_FOLDER_DEFAULTS) == len(IO_FOLDER_PURPUSES)


def parse_and_create():
    my_args = parser.parse_args()
    freshly_create_all_folders()
    # copy example inputs if option selected
    if my_args.examples:
        copy_examples()


def freshly_create_all_folders():
    for path in IO_FOLDER_DEFAULTS:
        if not os.path.exists(path):
            os.makedirs(path)
    with open(PATH_USER_PATHS, "w") as f:
        for varname, new_value in zip(IO_VARIABLE_NAMES, IO_FOLDER_DEFAULTS):
            f.write(f"{varname} = '{new_value}'\n")


def copy_examples():
    src_files = os.listdir(PATH_EXAMPLES)
    for file_name in src_files:
        full_file_name = os.path.join(PATH_EXAMPLES, file_name)
        if os.path.isfile(full_file_name):
            # copy PT examples to PATH_OUTPUT_PT
            if (full_file_name.endswith(".gro") and "ico" in full_file_name) or full_file_name.endswith(".xtc"):
                shutil.copy(full_file_name, IO_FOLDER_DEFAULTS[1])
            # copy the rest of the files to PATH_INPUT_BASEGRO
            elif full_file_name.endswith(".gro") or full_file_name.endswith(".pdb") or full_file_name.endswith(".xvg") \
                    or full_file_name.endswith(".xyz"):
                shutil.copy(full_file_name, IO_FOLDER_DEFAULTS[2])


if __name__ == '__main__':
    parse_and_create()
