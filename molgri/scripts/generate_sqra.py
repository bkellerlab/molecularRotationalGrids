"""
User script for generating either SQRA or MSM matrices and calculating their eigenvectors/eigenvalues.
"""

import argparse
from time import time
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt


from molgri.molecules.transitions import SimulationHistogram, SQRA
from molgri.plotting.transition_plots import TransitionPlot
from molgri.space.fullgrid import FullGrid
from molgri.scripts.set_up_io import freshly_create_all_folders

import warnings

from molgri.space.utils import k_argmax_in_array

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
parser.add_argument('-origingrid', metavar='og', type=str, nargs='?',
                           help='name of the rotation grid for rotations around origin in the form '
                                'algorithm_N (eg. ico_50) OR just a number (eg. 50)')
parser.add_argument('-bodygrid', metavar='bg', type=str, nargs='?',
                           help='name of the rotation grid for rotations around body in the form '
                                'algorithm_N (eg. ico_50) OR just a number (eg. 50) '
                                'OR None if you only want rotations about origin')
parser.add_argument('-transgrid', metavar='tg', type=str, nargs='?',
                           help='translation grid provided as a list of distances, as linspace(start, stop, num) '
                                'or range(start, stop, step) in nanometers')
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')
parser.add_argument('--save_as', type=str, default=None, help='define the (base) name of output file')
parser.add_argument('-energy', type=str, nargs='?',
                           help='name of the .xvg file (or other energy file) if not the same as the name of (pseudo)trajectory')

def run_generate_sqra():
    t1 = time()
    freshly_create_all_folders()
    my_args = parser.parse_args()

    use_saved = ~my_args.recalculate

    if my_args.bodygrid and my_args.origingrid and my_args.transgrid:
        fg = FullGrid(b_grid_name=my_args.bodygrid, o_grid_name=my_args.origingrid,
                              t_grid_name=my_args.transgrid, use_saved=use_saved)
    else:
        fg = None
    sh = SimulationHistogram(trajectory_name=my_args.pseudotrajectory, reference_name=my_args.m2, is_pt=True,
                             full_grid=fg,
                             second_molecule_selection=my_args.selection, use_saved=use_saved)


    sqra = SQRA(sh, use_saved=use_saved)

    sqra.get_transitions_matrix()
    eigenval, eigenvec = sqra.get_eigenval_eigenvec()

    sqra_tp = TransitionPlot((sh, sqra))
    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(10, 5))
    sqra_tp.plot_its(6, as_line=True, save=False, fig=fig, ax=ax[1])
    sqra_tp.plot_eigenvalues(num_eigenv=6, save=True, fig=fig, ax=ax[0])
    # x-values are irrelevant, they are just horizontal lines
    ax[1].set_xlabel("")
    ax[1].set_xticks([])

    fig, ax = plt.subplots(5, sharex=True, sharey=True, figsize=(5, 12.5))
    save=False
    for i in range(5):
        if i==4:
            save = True
        sqra_tp.plot_one_eigenvector_flat(i, save=save, fig=fig, ax=ax[i])

    print("You can use the following lists of cells to visualize eigenvectors in VMD")
    num_extremes = 30
    magnitudes = eigenvec[0].T[0]
    most_positive = k_argmax_in_array(np.abs(magnitudes), num_extremes)
    print(f"In 0th eigenvector {num_extremes} most positive cells are {list(most_positive+1)}.")
    for i in range(1, 5):
        magnitudes = eigenvec[0].T[i]
        most_positive = k_argmax_in_array(magnitudes, num_extremes)
        most_negative = k_argmax_in_array(-magnitudes, num_extremes)

        print(
            f"In {i}. eigenvector {num_extremes} most positive cells are {list(most_positive + 1)} and most negative {list(most_negative + 1)}.")

    t2 = time()
    print(f"Total time needed: ", end="")
    print(f"{timedelta(seconds=t2 - t1)} hours:minutes:seconds")


if __name__ == '__main__':
    run_generate_sqra()

