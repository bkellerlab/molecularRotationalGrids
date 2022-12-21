import argparse
import ast

import numpy as np

from molgri.paths import PATH_INPUT_ENERGIES
from molgri.scripts.set_up_io import freshly_create_all_folders
from molgri.plotting import EnergyConvergencePlot, create_trajectory_energy_multiplot

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-xvg', type=str, nargs='?', required=True,
                           help=f'name of the .xvg file in the {PATH_INPUT_ENERGIES} folder, extension not necessary')
parser.add_argument('--Ns_o', type=str, default=None,
                    help='a list of Ns <= number of orientational rotations')
parser.add_argument('--p1d', action='store_true', default=False,
                    help="construct 1-dimensional violin plots to help determine convergence")
parser.add_argument('--p3d', action='store_true', default=False,
                    help="construct 3-dimensional grid plots to help determine convergence")
parser.add_argument('--animate', action='store_true', default=False,
                    help="animate rotation of 3d plots, recommended (but may take a few minutes)")
parser.add_argument('--p2d', action='store_true', default=False,
                    help="construct 2-dimensional Hammer projection plots to help determine convergence")
parser.add_argument('--label', default="Potential",
                    help="one of y labels found in the XVG file (eg Potential), if not found, use the first y column")


def run_generate_pt():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    if my_args.Ns_o is None:
        Ns = None
    else:
        Ns_list = ast.literal_eval(my_args.Ns_o)
        Ns = [int(N) for N in Ns_list]
        Ns = np.array(Ns, dtype=int)
    if my_args.p1d:
        EnergyConvergencePlot(my_args.xvg, test_Ns=Ns, property_name=my_args.label).create_and_save()
    if my_args.p2d:
        print("Warning! Hammer plots not yet implemented. The flag --p2d has no effect.")
    if my_args.p3d:
        create_trajectory_energy_multiplot(my_args.xvg, Ns=Ns, animate_rot=my_args.animate)
    if my_args.animate and not my_args.p3d:
        print("Warning! No animation possible since no 3D plot is constructed. Use --p3d --animate if you "
              "want an animation of 3D energy distribution.")
    if not my_args.p1d and not my_args.p2d and not my_args.p3d:
        print("Select at least one of plotting options: --p1d, --p2d, --p3d")


if __name__ == '__main__':
    run_generate_pt()

