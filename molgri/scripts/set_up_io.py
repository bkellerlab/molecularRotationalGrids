"""
This script should be run to define input and output directories if the user wants to select directories other
than default ones. It can also be run at any later point to move the contents from previously selected directories
to new ones.
"""

# TODO: check for validity of inputs

import os
from os.path import samefile
from pathlib import Path
import shutil
from pkg_resources import resource_filename

from ..constants import PATH_USER_PATHS

# if you introduce new IO folders, add them to next three lines
IO_FOLDER_PURPUSES = ["rotational grids", "pseudotrajectory files", "base gro files", "plots", "animations",
                      "grid statistics"]
IO_FOLDER_DEFAULTS = ["output/grid_files/", "output/pt_files/", "input/base_gro_files/", "output/figures/",
                      "output/animations/", "output/statistics_files"]
#IO_FOLDER_DEFAULTS = [resource_filename("molgri", rel_path) for rel_path in IO_FOLDER_DEFAULTS]
IO_VARIABLE_NAMES = ["PATH_OUTPUT_ROTGRIDS", "PATH_OUTPUT_PT", "PATH_INPUT_BASEGRO", "PATH_OUTPUT_PLOTS",
                     "PATH_OUTPUT_ANIS", "PATH_OUTPUT_STAT"]

# which files are needed? Rotation grids (output), Pseudotrajectories (output), single-molecule gro files (input)


def freshly_create_all_folders():
    for path in IO_FOLDER_DEFAULTS:
        if not os.path.exists(path):
            os.makedirs(path)
    with open(PATH_USER_PATHS, "w") as f:
        for varname, new_value in zip(IO_VARIABLE_NAMES, IO_FOLDER_DEFAULTS):
            f.write(f"{varname} = '{new_value}'\n")


def detect_current_paths() -> list:
    """
    Reads the file defined by PATH_USER_PATHS where the current paths to I/O directories are saved and returns the
    paths as a list.

    Returns:
        a list of paths [path_to_rot_grids, path_to_pseudotrajectories, path_to_base_gro_files]
    """
    all_paths = []
    with open(PATH_USER_PATHS, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            keyword, path = line.split("=")
            path = path.strip()
            path = path.strip("'")
            all_paths.append(path)
    return all_paths


def create_io_folders(new_paths: list):
    """
    Rewrites the saved paths in PATH_USER_PATHS file by paths given by inputs.
    Creates new folders at paths given by inputs.
    Adds new paths to .gitignore file.

    Args:
        new_paths: list of paths in the same order as default
    """
    assert len(new_paths) == len(IO_FOLDER_DEFAULTS)
    # write down the new paths in paths.py file
    with open(PATH_USER_PATHS, "w") as f:
        for varname, new_value in zip(IO_VARIABLE_NAMES, new_paths):
            f.write(f"{varname} = '{new_value}'\n")
    # create the folders (if they do not exist yet)
    for path in new_paths:
        if not os.path.exists(path):
            os.makedirs(path)
    # # add them to .gitignore
    # with open(f".gitignore", "r") as f:
    #     lines = f.readlines()
    # with open(f".gitignore", "a") as f:
    #     for path in new_paths:
    #         if f"{path}\n" not in lines:
    #             f.write(f"{path}\n")


def move_and_delete_io_folders(current_folder: str, new_folder: str):
    """
    Moves all the files from the current folder to a new folder and deletes the (now empty) current folder.
    Assumes both folders exist. Also deletes reference to old folder from .gitignore if it exists.

    Args:
        current_folder: path to a starting folder
        new_folder: path to a new folder
    """
    # moves files to a new location
    for current_file in Path(current_folder).glob('*.*'):
        shutil.copy(current_file, new_folder)
    # deletes old folders and files in it
    shutil.rmtree(current_folder)
    # # delete old folder from .gitignore
    # with open(f".gitignore", "r") as f:
    #     lines = f.readlines()
    # with open(f".gitignore", "w") as f:
    #     for line in lines:
    #         if line.strip("\n") != current_folder:
    #             f.write(line)


def run_user_input_program():
    """
    This script interacts with the user and asks for new paths to I/O files.
    """
    current_values = detect_current_paths()
    user_values = []
    print(f"Default i/o folders are: {IO_FOLDER_DEFAULTS}.")
    reset = input("Reset all i/o folders to default values? (y/n) ")
    if reset.strip() in ["y", "yes"]:
        user_values = IO_FOLDER_DEFAULTS
    else:
        for message, default in zip(IO_FOLDER_PURPUSES, current_values):
            change1 = f"Current folder for saving {message} is {default}. Do you want to change it? (y/n) "
            wanna_change1 = input(change1)
            if wanna_change1.strip() in ["y", "yes"]:
                message1 = f"Select a new folder where {message} will be saved: "
                user_selection = input(message1)
                user_values.append(user_selection)
            else:
                user_values.append(default)
    # create new folders and references
    create_io_folders(user_values)
    # Ask if you should move & delete from previous folders
    for ov, nv in zip(current_values, user_values):
        if not samefile(ov, nv):
            move = input(f"Move files from previous {ov} folder to the new {nv} folder and delete old folder? (y/n) ")
            if move.strip() in ["y", "yes"]:
                # Move contents from current folders to new folders
                # Delete previous I/O folders
                move_and_delete_io_folders(ov, nv)


if __name__ == '__main__':
    #run_user_input_program()
    freshly_create_all_folders()
