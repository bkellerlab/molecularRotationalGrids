"""
This is a user-friendly script for generating a custom rotational grid.
"""

import os
from os.path import join
import argparse

from molgri.grids import Grid, build_grid
from molgri.parsers import NameParser
from ..paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS, PATH_OUTPUT_STAT
from ..constants import ENDING_GRID_FILES
from molgri.plotting import GridPlot, AlphaViolinPlot, AlphaConvergencePlot
from ..scripts.set_up_io import freshly_create_all_folders

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-N', metavar='N', type=int, nargs='?', required=True,
                           help='the number of points in your rotational grid')
requiredNamed.add_argument('-algorithm', metavar='a', type=str, nargs='?', required=True,
                           help='name of the grid-generating algorithm to use'
                                '(ico, cube3D, cube4D, randomQ, randomE, systemE)')
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')
parser.add_argument('--statistics', action='store_true',
                    help='write out statistics and draw uniformity and convergence plots')
parser.add_argument('--draw', action='store_true',
                    help='draw the grid and save the plot')
parser.add_argument('--background', action='store_true',
                    help='when drawing a grid, display axes and ticks.')
parser.add_argument('--animate', action='store_true',
                    help='provide an animation of the grid rotating in 3D')
parser.add_argument('--animate_ordering', action='store_true',
                    help='provide an animation of the grid generation')
parser.add_argument('--readable', action='store_true',
                    help='save the grid file in a txt format as well')


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
    # if running from another script, args may not include the readable attribute
    try:
        if args.readable:
            my_grid.save_grid_txt()
            print(f"Saved a human-readable version of rotation grid to {PATH_OUTPUT_ROTGRIDS}{name}.txt")
    except AttributeError:
        pass
    return my_grid


def run_generate_grid():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    nap = NameParser(f"{my_args.algorithm}_{my_args.N}")
    grid_name = nap.get_standard_name()
    my_grid = prepare_grid(my_args, nap)
    if my_args.animate or my_args.animate_ordering:
        my_args.draw = True
    if my_args.draw:
        if my_args.background:
            style_type = ["talk"]
        else:
            style_type = ["talk", "empty"]
        my_gp = GridPlot(grid_name, style_type=style_type)
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
        my_grid.save_statistics(print_message=True)
        print(f"A statistics file describing the grid {grid_name} was saved to {PATH_OUTPUT_STAT}.")
        # create violin plot
        AlphaViolinPlot(grid_name).create()
        print(f"A violin plot showing the uniformity of {grid_name} saved to {PATH_OUTPUT_PLOTS}.")
        # create covergence plot
        AlphaConvergencePlot(grid_name).create()
        print(f"A convergence plot with number of points between 3 and {nap.num_grid_points} saved "
              f"to {PATH_OUTPUT_PLOTS}.")


if __name__ == '__main__':
    run_generate_grid()
