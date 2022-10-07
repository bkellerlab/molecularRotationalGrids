"""
This is a user-friendly script for generating a custom rotational grid.
"""

import os
from os.path import join
import argparse

from molgri.grids.grid import build_grid, Grid
from molgri.parsers.name_parser import NameParser
from molgri.paths import PATH_OUTPUT_ROTGRIDS, PATH_OUTPUT_GRIDPLOT, PATH_OUTPUT_GRIDORDER_ANI, PATH_OUTPUT_GRID_ANI
from molgri.my_constants import ENDING_GRID_FILES
from molgri.plotting.plot_grids import GridPlot

parser = argparse.ArgumentParser()
parser.add_argument('N', metavar='N', type=int, nargs='?',
                    default=500, help='Number of points per rotational grid.')
parser.add_argument('--algorithm', metavar='g', type=str, nargs='?',
                    default='ico', help='Which grid-generating algorithm to use?\
                    (ico, cube3D, cube4D, randomQ, randomE, systemE)')
parser.add_argument('--recalculate', action='store_true',
                    help='Even if a saved version of this grid already exists, recalculate it.')
parser.add_argument('--statistics', action='store_true',
                    help='Write out statistics about this grid.')
parser.add_argument('--draw', action='store_true',
                    help='Draw this grid and display a figure.')
parser.add_argument('--animate', action='store_true',
                    help='Provide an animation of the grid display.')
parser.add_argument('--animate_ordering', action='store_true',
                    help='Provide an animation of the grid ordering.')
# TODO --statistics


def prepare_grid(args, name) -> Grid:
    # if already exists and no --recalculate flag, just display a message
    if os.path.exists(join(PATH_OUTPUT_ROTGRIDS, f"{grid_name}.{ENDING_GRID_FILES}")) and not args.recalculate:
        print(f"Grid with name {name} is already saved. If you want to recalculate it, select --recalculate flag.")
        my_grid = build_grid(args.grid_type, args.N, use_saved=True)
    else:
        my_grid = build_grid(args.grid_type, args.N, use_saved=False)
        my_grid.generate_and_time(print_message=True)
        my_grid.save_grid()
    return my_grid


if __name__ == '__main__':
    my_args = parser.parse_args()
    grid_name = NameParser(f"{my_args.grid_type}_{my_args.N}").get_standard_name()
    prepare_grid(my_args, grid_name)
    if my_args.animate or my_args.animate_ordering:
        my_args.draw = True
    if my_args.draw:
        my_gp = GridPlot(grid_name, empty=True, title=False)
        my_gp.create()
        print(f"Grid drawn and figure saved to {PATH_OUTPUT_GRIDPLOT}.")
        if my_args.animate:
            my_gp.animate_figure_view()
            print(f"Animation of the grid saved to {PATH_OUTPUT_GRID_ANI}")
        if my_args.animate_ordering:
            my_gp.animate_grid_sequence()
            print(f"Animation of the grid ordering saved to {PATH_OUTPUT_GRIDORDER_ANI}")
