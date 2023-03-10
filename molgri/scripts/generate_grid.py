"""
This is a user-friendly script for generating a custom rotational grid.
"""

import argparse

from molgri.space.rotobj import SphereGridFactory
from molgri.plotting.spheregrid_plots import SphereGridPlot
from molgri.scripts.set_up_io import freshly_create_all_folders

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-N', metavar='N', type=int, nargs='?', required=True,
                           help='the number of points in your rotational grid')
requiredNamed.add_argument('-algorithm', metavar='a', type=str, nargs='?', required=True,
                           help='name of the grid-generating algorithm to use'
                                ' (ico, cube3D, cube4D, randomQ, randomE, systemE)')
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')
parser.add_argument('--dimensions', type=int, default=3,
                    help='select 3 for grids on a sphere in 3D space and 4 for grids on a hypersphere in 4D space')
parser.add_argument('--statistics', action='store_true',
                    help='write out statistics and draw uniformity and convergence plots')
parser.add_argument('--draw', action='store_true',
                    help='draw the grid and save the plot')
parser.add_argument('--background', action='store_true', default=True,
                    help='when drawing a grid, display axes and ticks.')
parser.add_argument('--animate', action='store_true',
                    help='provide an animation of the grid rotating in 3D')
parser.add_argument('--animate_ordering', action='store_true',
                    help='provide an animation of the grid generation')
parser.add_argument('--animate_translation', action='store_true',
                    help='provide an animation of the grid sliding through one of the dimensions')
parser.add_argument('--readable', action='store_true',
                    help='save the grid file in a txt format as well')


def prepare_grid(args):
    algo = args.algorithm
    n_points = args.N
    # if already exists and no --recalculate flag, just display a message
    use_saved = not args.recalculate
    my_sphere_grid = SphereGridFactory.create(N=n_points, alg_name=algo, dimensions=args.dimensions,
                                              use_saved=use_saved, time_generation=True, print_messages=True)
    my_sphere_grid.save_grid()
    print(f"The grid can be found at {my_sphere_grid.get_grid_path()}")
    # if running from another script, args may not include the readable attribute
    try:
        if args.readable:
            extension = "txt"
            my_sphere_grid.save_grid(extension=extension)
            path = my_sphere_grid.get_grid_path(extension=extension)
            print(f"A .txt version of grid can be found at {path}.")
    except AttributeError:
        pass
    return my_sphere_grid


def run_generate_grid():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    grid_name = f"{my_args.algorithm}_{my_args.N}"
    my_rotations = prepare_grid(my_args)

    # any plotting is done with this object
    default_complexity_level = "full"
    if not my_args.background:
        default_complexity_level = "empty"
    sgp = SphereGridPlot(my_rotations, default_context="talk", default_complexity_level=default_complexity_level)

    if my_args.animate or my_args.animate_ordering:
        my_args.draw = True
    if my_args.draw:
        sgp.make_grid_plot()
        print(f"Grid drawn and figure saved to {sgp.fig_path}.")
        if my_args.animate:
            sgp.make_rot_animation()
            print(f"Animation of the grid saved to {sgp.ani_path}")
        if my_args.animate_ordering:
            sgp.make_ordering_animation()
            print(f"Animation of the grid ordering saved to {sgp.ani_path}")
        if my_args.animate_translation:
            sgp.make_trans_animation()
            print(f"Animation of the grid sliding through the last dimension saved to {sgp.ani_path}")
    if my_args.statistics:
        my_rotations.save_uniformity_statistics()
        print(f"A statistics file describing the grid {grid_name} was saved to {my_rotations.get_statistics_path('csv')}.")
        sgp.make_uniformity_plot()
        print(f"A violin plot showing the uniformity of {grid_name} saved to {sgp.fig_path}.")
        sgp.make_convergence_plot()
        print(f"A convergence plot with number of points between 3 and {my_args.N} saved to {sgp.fig_path}.")


if __name__ == '__main__':
    run_generate_grid()
    print("Generation finished.")
