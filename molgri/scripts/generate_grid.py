"""
This is a user-friendly script for generating a custom rotational grid.
"""

import os
from os.path import join
import argparse

from molgri.grids import Grid, build_grid
from molgri.parsers import NameParser
from ..paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS
from ..constants import ENDING_GRID_FILES
from molgri.plotting import GridPlot, AlphaViolinPlot, AlphaConvergencePlot
from ..scripts.set_up_io import freshly_create_all_folders

# TODO --format allow to save also as readable file

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-N', metavar='N', type=int, nargs='?', required=True,
                           help='Number of points per rotational grid.')
requiredNamed.add_argument('-algorithm', metavar='a', type=str, nargs='?', required=True,
                           help='Which grid-generating algorithm to use?\
                    (ico, cube3D, cube4D, randomQ, randomE, systemE)')
parser.add_argument('--recalculate', action='store_true',
                    help='Even if a saved version of this grid already exists, recalculate it.')
parser.add_argument('--statistics', action='store_true',
                    help='Write out statistics and draw statistics plots about this grid.')
parser.add_argument('--draw', action='store_true',
                    help='Draw this grid and display a figure.')
parser.add_argument('--animate', action='store_true',
                    help='Provide an animation of the grid display.')
parser.add_argument('--animate_ordering', action='store_true',
                    help='Provide an animation of the grid ordering.')
parser.add_argument('--readable', action='store_true',
                    help='Also save the grid in a human-readable.txt format.')


def prepare_grid(args, parsed_name: NameParser) -> Grid:
    name = parsed_name.get_standard_name()
    algo = parsed_name.grid_type
    n_points = parsed_name.num_grid_points
    # if already exists and no --recalculate flag, just display a message
    if os.path.exists(join(PATH_OUTPUT_ROTGRIDS, f"{name}.{ENDING_GRID_FILES}")) and not args.recalculate:
        print(f"Grid with name {name} is already saved. If you want to recalculate it, select --recalculate flag.")
        my_grid = build_grid(algo, n_points, use_saved=True, time_generation=True)
    else:
        my_grid = build_grid(algo, n_points, use_saved=False, time_generation=True)
        my_grid.save_grid()
        print(f"Generated a {my_grid.decorator_label} with {my_grid.N} points.")
    if args.readable:
        my_grid.save_grid_txt()
        print(f"Saved a human-readable version of rotation grid to {PATH_OUTPUT_ROTGRIDS}{name}.txt")
    return my_grid


def run_generate_grid():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    nap = NameParser(f"{my_args.algorithm}_{my_args.N}")
    grid_name = nap.get_standard_name()
    prepare_grid(my_args, nap)
    if my_args.animate or my_args.animate_ordering:
        my_args.draw = True
    if my_args.draw:
        my_gp = GridPlot(grid_name, style_type=["talk", "empty"])
        my_gp.create()
        print(f"Grid drawn and figure saved to {PATH_OUTPUT_PLOTS}.")
        if my_args.animate:
            my_gp.animate_figure_view()
            print(f"Animation of the grid saved to {PATH_OUTPUT_ANIS}")
        if my_args.animate_ordering:
            my_gp.animate_grid_sequence()
            print(f"Animation of the grid ordering saved to {PATH_OUTPUT_ANIS}")
    if my_args.statistics:
        # create statistic data
        # create violin plot
        AlphaViolinPlot(grid_name).create()
        print(f"A violin plot showing the uniformity of {grid_name} saved to {PATH_OUTPUT_PLOTS}.")
        # create covergence plot
        AlphaConvergencePlot(grid_name).create()
        print(f"A convergence plot with number of points between 3 and {nap.num_grid_points} saved "
              f"to {PATH_OUTPUT_PLOTS}.")

if __name__ == '__main__':
    run_generate_grid()
