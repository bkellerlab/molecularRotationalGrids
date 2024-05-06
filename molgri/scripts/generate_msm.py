"""
User script for generating either SQRA or MSM matrices and calculating their eigenvectors/eigenvalues.
"""

import argparse

import numpy as np

from molgri.constants import TAUS
from molgri.molecules.transitions import SimulationHistogram, MSM, SQRA
from molgri.plotting.transition_plots import TransitionPlot
from molgri.space.fullgrid import FullGrid
from molgri.scripts.set_up_io import freshly_create_all_folders

import warnings
warnings.filterwarnings("ignore")

# TODO: define total_N and generate in all dimensions uniform grid?

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-m2', type=str, nargs='?', required=True,
                           help='name of the .gro file (or other structure file) of the moving molecule')
requiredNamed.add_argument('-pseudotrajectory', type=str, nargs='?', required=True,
                           help='name of the .xtc file (or other trajectory file) containing (pseudo)trajectory')

requiredNamed.add_argument('-selection', type=str, nargs='?', required=True,
                           help='select which particles belong to the moving molecule using MDAnalysis selection commands')
requiredNamed.add_argument('-origingrid', metavar='og', type=str, nargs='?', required=True,
                           help='name of the rotation grid for rotations around origin in the form '
                                'algorithm_N (eg. ico_50) OR just a number (eg. 50)')
requiredNamed.add_argument('-bodygrid', metavar='bg', type=str, nargs='?', required=True,
                           help='name of the rotation grid for rotations around body in the form '
                                'algorithm_N (eg. ico_50) OR just a number (eg. 50) '
                                'OR None if you only want rotations about origin')
requiredNamed.add_argument('-transgrid', metavar='tg', type=str, nargs='?', required=True,
                           help='translation grid provided as a list of distances, as linspace(start, stop, num) '
                                'or range(start, stop, step) in nanometers')
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')
parser.add_argument('--save_as', type=str, default=None, help='define the (base) name of output file')
parser.add_argument('--msm', action='store_true', help='use MSM (default is SQRA)')
parser.add_argument('--taus', type=str, default=str(TAUS), help='give a list of tau values to use')
parser.add_argument('-energy', type=str, nargs='?',
                           help='name of the .xvg file (or other energy file) if not the same as the name of (pseudo)trajectory')

def run_generate_msm():
    freshly_create_all_folders()
    my_args = parser.parse_args()

    use_saved = ~my_args.recalculate

    fg = FullGrid(b_grid_name=my_args.bodygrid, o_grid_name=my_args.origingrid,
                          t_grid_name=my_args.transgrid, use_saved=use_saved)
    sh = SimulationHistogram(trajectory_name=my_args.pseudotrajectory, reference_name=my_args.m2, is_pt=~my_args.msm, full_grid=fg,
                             second_molecule_selection=my_args.selection, use_saved=use_saved)

    if my_args.msm:
        transition_model = MSM(sh, tau_array=np.array(my_args.taus), use_saved=use_saved)
    else:
        transition_model = SQRA(sh, use_saved=use_saved)

    transition_model.get_transitions_matrix()
    transition_model.get_eigenval_eigenvec()

if __name__ == '__main__':
    run_generate_msm()

