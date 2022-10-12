"""
This script should be run to define input and output directories if the user wants to select directories other
than default ones. It can also be run at any later point to move the contents from previously selected directories
to new ones.
"""

import os

from ..constants import PATH_USER_PATHS

# if you introduce new IO folders, add them to next three lines
IO_FOLDER_PURPUSES = ["rotational grids", "pseudotrajectory files", "base gro files", "plots", "animations",
                      "grid statistics"]
IO_FOLDER_DEFAULTS = ["output/grid_files/", "output/pt_files/", "input/base_gro_files/", "output/figures/",
                      "output/animations/", "output/statistics_files"]
IO_VARIABLE_NAMES = ["PATH_OUTPUT_ROTGRIDS", "PATH_OUTPUT_PT", "PATH_INPUT_BASEGRO", "PATH_OUTPUT_PLOTS",
                     "PATH_OUTPUT_ANIS", "PATH_OUTPUT_STAT"]


def freshly_create_all_folders():
    for path in IO_FOLDER_DEFAULTS:
        if not os.path.exists(path):
            os.makedirs(path)
    with open(PATH_USER_PATHS, "w") as f:
        for varname, new_value in zip(IO_VARIABLE_NAMES, IO_FOLDER_DEFAULTS):
            f.write(f"{varname} = '{new_value}'\n")


if __name__ == '__main__':
    freshly_create_all_folders()
