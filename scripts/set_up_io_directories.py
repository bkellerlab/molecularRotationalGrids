"""
This script should be run first to define input and output directories.
"""

import sys
import os

from ..molgri.my_constants import PATH_USER_PATHS


# which files are needed? Rotation grids (output), Pseudotrajectories (output), single-molecule gro files (input)
# paths.py saves the here-defined paths


def create_io_folders(rot_grids_path="output/grid_files/",
                      pt_path="output/pt_files/",
                      base_gro_path="input/base_gro_files"):
    # write down the requested folders in paths.py file
    with open(PATH_USER_PATHS, "w") as f:
        f.write(f"PATH_OUTPUT_ROTGRIDS = {rot_grids_path}\n")
        f.write(f"PATH_OUTPUT_PT = {pt_path}\n")
        f.write(f"PATH_INPUT_BASEGRO = {base_gro_path}\n")
    # create the folders

def delete_io_folders():
    pass


user_input = input("Enter the path of your file: ")

assert os.path.exists(user_input), "I did not find the file at, " + str(user_input)
f = open(user_input, 'r+')
print("Hooray we found your file!")
# stuff you do with the file goes here
f.close()