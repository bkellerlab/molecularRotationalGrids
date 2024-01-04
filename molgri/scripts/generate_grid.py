"""
User script for generating custom rotational grids. Quickly get rotobj and their
- statistics
- plots/animations
- voronoi cells/adjacencies/surfaces ..
"""

import argparse

from molgri.constants import DEFAULT_ALGORITHM_B, DEFAULT_ALGORITHM_O

from molgri.space.rotobj import SphereGrid3DFactory, SphereGrid4DFactory
from molgri.plotting.spheregrid_plots import SphereGridPlot
from molgri.plotting.voronoi_plots import VoronoiPlot
from molgri.scripts.set_up_io import freshly_create_all_folders

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-N', metavar='N', type=int, nargs='?', required=True,
                           help='the number of points in your rotational grid')
requiredNamed.add_argument('-d', metavar='d', type=int, nargs='?', required=True,
                           help='the number of dimensions (3, 4) in which you are creating a grid')
parser.add_argument('--algorithm', type=str, default=None,
                    help='define an algorithm that you want to use if different from default')
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')
parser.add_argument('--statistics', action='store_true',
                    help='write out statistics and draw uniformity and convergence plots')
parser.add_argument('--draw', action='store_true',
                    help='draw the grid and save the plot')
parser.add_argument('--animate', action='store_true',
                    help='provide an animation of the grid rotating in 3D')
parser.add_argument('--animate_ordering', action='store_true',
                    help='provide an animation of the grid generation')
parser.add_argument('--animate_translation', action='store_true',
                    help='provide an animation of the grid sliding through one of the dimensions')


def run_generate_grid():
    freshly_create_all_folders()
    my_args = parser.parse_args()

    # if already exists and no --recalculate flag, just display a message
    use_saved = not my_args.recalculate

    # check dimensions and create accordingly; also create the belonging voronoi object
    if my_args.d == 3:
        if my_args.algorithm is None:
            my_args.algorithm = DEFAULT_ALGORITHM_O
        my_factory = SphereGrid3DFactory
    elif my_args.d == 4:
        if my_args.algorithm is None:
            my_args.algorithm = DEFAULT_ALGORITHM_B
        my_factory = SphereGrid4DFactory
    else:
        raise ValueError(f"Cannot create a grid with {my_args.d} dimensions, try d=3 or d=4.")

    my_rotations = my_factory.create(N=my_args.N, alg_name=my_args.algorithm, use_saved=use_saved, time_generation=True)
    grid_name = my_rotations.get_name(with_dim=True)

    # any plotting is done with this object
    default_complexity_level = "half_empty"
    sgp = SphereGridPlot(my_rotations, default_context="talk", default_complexity_level=default_complexity_level)

    if my_args.animate or my_args.animate_ordering:
        my_args.draw = True
    if my_args.draw:
        sgp.plot_grid(animate_rot=my_args.animate)
        if my_args.N > 4:
            # and here the voronoi plotting
            vp = VoronoiPlot(my_rotations.get_spherical_voronoi(), default_context="talk",
                             default_complexity_level=default_complexity_level)
            vp.create_all_plots(and_animations=my_args.animate)
        print(f"Grid drawn and figure saved to {sgp.fig_path}.")
        if my_args.animate:
            print(f"Animation of the grid saved to {sgp.ani_path}")
        if my_args.animate_ordering:
            sgp.animate_ordering()
            print(f"Animation of the grid ordering saved to {sgp.ani_path}")
        if my_args.animate_translation:
            sgp.animate_translation()
            print(f"Animation of the grid sliding through the last dimension saved to {sgp.ani_path}")
    if my_args.statistics:
        my_rotations.save_uniformity_statistics()
        print(f"A statistics file describing the grid {grid_name} was saved to {my_rotations.get_statistics_path('csv')}.")
        sgp.plot_uniformity()
        print(f"A violin plot showing the uniformity of {grid_name} saved to {sgp.fig_path}.")
        sgp.plot_convergence()
        print(f"A convergence plot with number of points between 3 and {my_args.N} saved to {sgp.fig_path}.")


if __name__ == '__main__':
    run_generate_grid()
    print("Generation finished.")
