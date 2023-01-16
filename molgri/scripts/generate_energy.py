import argparse
import ast

import numpy as np

from molgri.paths import PATH_INPUT_ENERGIES
from molgri.scripts.set_up_io import freshly_create_all_folders
from molgri.plotting.analysis_plots import EnergyConvergencePlot
from molgri.plotting.grid_plots import create_trajectory_energy_multiplot, create_hammer_multiplot, \
    HammerProjectionTrajectory, TrajectoryEnergyPlot

parser = argparse.ArgumentParser(description="""NOTE: This script is under development and likely to undergo changes in
the future. It is recommended to verify the plausibility of the visualisations. Feedback and feature requests are welcome.

Visualisation types:
 --p1d is a 1-dimensional violin plot that displays which energy values occur in the provided .xvg file and how
       common they are. It is a low-dimensional projection, so information is lost, but is a fast and highly simplified
       plot that can be useful for complex systems, especially when checking for convergence.
 --p2d is a 2D projection where the center of mass of the second molecule is displayed as a point in Hammer projection.
   The color of the point indicates the minimal energy at this point among all orientations tested (assuming body rotations
   were used in the pseudotrajectory).
 --p3d is a 3D display of points, determined and colored in the same way as in the p2d example. Instead of a point,
   the entire voranoi cell corresponding to it is colored. Should be used with the --animate flag since only an animation
   can properly display 3D information
""")
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-xvg', type=str, nargs='?', required=True,
                           help=f'name of the .xvg file in the {PATH_INPUT_ENERGIES} folder, extension not necessary')
parser.add_argument('--Ns_o', type=str, default=None,
                    help='a list of Ns <= number of orientational rotations')
parser.add_argument('--p1d', action='store_true', default=False,
                    help="construct 1-dimensional violin plots to help determine convergence")
parser.add_argument('--p3d', action='store_true', default=False,
                    help="construct 3-dimensional grid plots colored in accordance to minimal energy per point")
parser.add_argument('--animate', action='store_true', default=False,
                    help="animate rotation of 3d plots, recommended (but may take a few minutes)")
parser.add_argument('--p2d', action='store_true', default=False,
                    help="construct 2-dimensional Hammer projection plots colored in accordance to minimal energy per point")
parser.add_argument('--label', default="Potential",
                    help="one of y labels found in the XVG file (eg Potential), if not found, use the first y column")
parser.add_argument('--convergence', action='store_true', default=False,
                    help="select if you want to produce series of plots at different Ns (that you can select"
                         "with flag --Ns_o")


def run_generate_energy():
    freshly_create_all_folders()
    my_args = parser.parse_args()
    if my_args.Ns_o is None:
        Ns = None
    else:
        Ns_list = ast.literal_eval(my_args.Ns_o)
        Ns = [int(N) for N in Ns_list]
        Ns = np.array(Ns, dtype=int)
    # remove the xvg ending if necessary
    if my_args.xvg.endswith(".xvg"):
        data_name = my_args.xvg[:-4]
    else:
        data_name = my_args.xvg
    if my_args.p1d:
        if my_args.convergence:
            EnergyConvergencePlot(data_name, test_Ns=Ns, property_name=my_args.label).create_and_save()
        else:
            EnergyConvergencePlot(data_name, no_convergence=True, property_name=my_args.label).create_and_save()
    if my_args.p2d:
        if my_args.convergence:
            create_hammer_multiplot(data_name, Ns=Ns)
        else:
            hpt = HammerProjectionTrajectory(data_name)
            hpt.add_energy_information(f"{PATH_INPUT_ENERGIES}{data_name}")
            hpt.create_and_save()
    if my_args.p3d:
        if my_args.convergence:
            create_trajectory_energy_multiplot(data_name, Ns=Ns, animate_rot=my_args.animate)
        else:
            tep = TrajectoryEnergyPlot(data_name, plot_points=False, plot_surfaces=True)
            tep.add_energy_information(f"{PATH_INPUT_ENERGIES}{data_name}")
            tep.create_and_save(animate_rot=my_args.animate)
    if my_args.animate and not my_args.p3d:
        print("Warning! No animation possible since no 3D plot is constructed. Use --p3d --animate if you "
              "want an animation of 3D energy distribution.")
    if not my_args.p1d and not my_args.p2d and not my_args.p3d:
        print("Select at least one of plotting options: --p1d, --p2d, --p3d")
    if my_args.Ns_o and not my_args.convergence:
        print("You have selected a list of test N values with flag --Ns_o. Do you want to test convergence? "
              "Please select an additional flag --convergence.")


if __name__ == '__main__':
    run_generate_energy()

