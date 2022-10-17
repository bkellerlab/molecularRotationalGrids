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
                      "grid statistics"]
IO_FOLDER_DEFAULTS = ["output/grid_files/", "output/pt_files/", "input/", "output/figures/",
                      "output/animations/", "output/statistics_files/"]
IO_VARIABLE_NAMES = ["PATH_OUTPUT_ROTGRIDS", "PATH_OUTPUT_PT", "PATH_INPUT_BASEGRO", "PATH_OUTPUT_PLOTS",
                     "PATH_OUTPUT_ANIS", "PATH_OUTPUT_STAT"]


def parse_and_create():
    my_args = parser.parse_args()
    freshly_create_all_folders()
    # copy example inputs if option selected
    if my_args.examples:
        src_files = os.listdir(PATH_EXAMPLES)
        for file_name in src_files:
            full_file_name = os.path.join(PATH_EXAMPLES, file_name)
            if os.path.isfile(full_file_name) and full_file_name.endswith(".gro"):
                shutil.copy(full_file_name, IO_FOLDER_DEFAULTS[2])


def freshly_create_all_folders():
    for path in IO_FOLDER_DEFAULTS:
        if not os.path.exists(path):
            os.makedirs(path)
    with open(PATH_USER_PATHS, "w") as f:
        for varname, new_value in zip(IO_VARIABLE_NAMES, IO_FOLDER_DEFAULTS):
            f.write(f"{varname} = '{new_value}'\n")


if __name__ == '__main__':
    parse_and_create()
