"""
This is a user-friendly script for generating a custom rotational grid.
"""

import os
from os.path import join
import argparse
from typing import Tuple

from molgri.space.rotobj import build_rotations
from ..paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_PLOTS, PATH_OUTPUT_ANIS, PATH_OUTPUT_STAT
from ..constants import EXTENSION_GRID_FILES
from molgri.plotting.grid_plots import GridPlot
from molgri.plotting.analysis_plots import AlphaViolinPlot, AlphaConvergencePlot, AlphaViolinPlotRot, \
    AlphaConvergencePlotRot
from ..scripts.set_up_io import freshly_create_all_folders

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-N', metavar='N', type=int, nargs='?', required=True,
                           help='the number of points in your rotational grid')
requiredNamed.add_argument('-algorithm', metavar='a', type=str, nargs='?', required=True,
                           help='name of the grid-generating algorithm to use'
                                ' (ico, cube3D, cube4D, randomQ, randomE, systemE)')
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


def prepare_grid(args, grid_name: str) -> Tuple:
    name = grid_name
    algo = args.algorithm
    n_points = args.N
    # if already exists and no --recalculate flag, just display a message
    use_saved = not args.recalculate
    my_rotations = build_rotations(n_points, algo, use_saved=use_saved, time_generation=True)
    if os.path.exists(join(PATH_OUTPUT_ROTGRIDS, f"{name}.{EXTENSION_GRID_FILES}")) and not args.recalculate:
        print(f"Grid with name {name} is already saved. If you want to recalculate it, select --recalculate flag.")
    my_grid = my_rotations.get_grid_z_as_grid()
    print(f"Generated a {my_rotations.decorator_label} with {my_rotations.N} points.")
    # if running from another script, args may not include the readable attribute
    try:
        if args.readable:
            my_grid.save_grid_txt()
            print(f"Saved a human-readable version of rotation grid to {PATH_OUTPUT_ROTGRIDS}{name}.txt")
    except AttributeError:
        pass
    return my_rotations, my_grid


def run_generate_grid():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    grid_name = f"{my_args.algorithm}_{my_args.N}"
    my_rotations, my_grid = prepare_grid(my_args, grid_name)
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
        my_rotations.save_statistics(print_message=True)
        my_grid.save_statistics(print_message=True)
        print(f"A statistics file describing the grid {grid_name} was saved to {PATH_OUTPUT_STAT}.")
        # create violin plot
        AlphaViolinPlot(grid_name).create_and_save()
        AlphaViolinPlotRot(grid_name).create_and_save()
        print(f"A violin plot showing the uniformity of {grid_name} saved to {PATH_OUTPUT_PLOTS}.")
        # create covergence plot
        AlphaConvergencePlot(grid_name).create_and_save()
        AlphaConvergencePlotRot(grid_name).create_and_save()
        print(f"A convergence plot with number of points between 3 and {my_args.N} saved "
              f"to {PATH_OUTPUT_PLOTS}.")


if __name__ == '__main__':
    run_generate_grid()
