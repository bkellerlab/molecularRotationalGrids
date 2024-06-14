"""
User script for generating  SQRA matrices and calculating their eigenvectors/eigenvalues.
"""

import argparse
from time import time
from datetime import timedelta

import numpy as np


from molgri.molecules.transitions import SimulationHistogram, SQRA
from molgri.paths import PATH_OUTPUT_AUTOSAVE
from molgri.space.fullgrid import FullGrid
from molgri.scripts.set_up_io import freshly_create_all_folders
from molgri.scripts.abstract_scripts import ScriptLogbook
from molgri.molecules.rate_merger import MatrixDecomposer

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
parser.add_argument('--merge_cutoff', metavar='tg', type=float, nargs='?',
                           help='Bottom cutoff in energy [k_BT], if difference below the cutoff, cells will be merged.',
                    default=0.01)
parser.add_argument('--cut_cutoff', metavar='tg', type=float, nargs='?',
                           help='Top cutoff in energy [k_BT], if energy above this, cell will be cut off.',
                    default=10)
parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')
parser.add_argument('--save_as', type=str, default=None, help='define the (base) name of output file')
parser.add_argument('-energy', type=str, nargs='?',
                           help='name of the .xvg file (or other energy file) if not the same as the name of (pseudo)trajectory')

def run_or_rerun():
    freshly_create_all_folders()
    my_args = parser.parse_args()

    use_saved = not my_args.recalculate

    my_log = ScriptLogbook(parser)
    print("MY LOG", my_log.my_index, my_log.is_newly_assigned)

    if use_saved and not my_log.is_newly_assigned:
        rerun_generate_sqra()
    else:
        run_generate_sqra(my_args, my_log, use_saved)

    with open(f"{PATH_OUTPUT_AUTOSAVE}{my_log.class_name}_{my_log.my_index}.txt", mode="r") as f:
        print(f.read())


def rerun_generate_sqra():
    print("PLEASE NOTE: This exact calculation has been done before. These are old saved results.", end=" ")
    print("If you do need a fresh calculation, use the option --recalculate")



def run_generate_sqra(my_args, my_log, use_saved):
    print_file = open(f"{PATH_OUTPUT_AUTOSAVE}{my_log.class_name}_{my_log.my_index}.txt", mode="w")
    t1 = time()

    if my_args.bodygrid and my_args.origingrid and my_args.transgrid:
        fg = FullGrid(b_grid_name=my_args.bodygrid, o_grid_name=my_args.origingrid,
                              t_grid_name=my_args.transgrid, use_saved=use_saved)
    else:
        fg = None
    sh = SimulationHistogram(trajectory_name=my_args.pseudotrajectory, reference_name=my_args.m2, is_pt=True,
                             full_grid=fg,
                             second_molecule_selection=my_args.selection, use_saved=use_saved)

    if my_args.origingrid is None:
        my_log.add_information({"origingrid": sh.full_grid.o_grid_name})
    if my_args.bodygrid is None:
        my_log.add_information({"bodygrid": sh.full_grid.b_grid_name})
    if my_args.transgrid is None:
        my_log.add_information({"transgrid": sh.full_grid.t_grid_name})

    sqra = SQRA(sh, use_saved=use_saved)

    rate_matrix = sqra.get_transitions_matrix()
    # cutting and merging
    print(f"################## ATTEMPTING TO REDUCE THE SIZE OF THE RATE MATRIX ##################",
          file=print_file)
    my_decomposer = MatrixDecomposer(rate_matrix, my_matrix_name=sqra.get_name(), use_saved=use_saved)
    size_pre_merge = my_decomposer.my_matrix.shape[0]
    my_decomposer.merge_my_matrix(sh, bottom_treshold=my_args.merge_cutoff)
    size_post_merge = my_decomposer.my_matrix.shape[0]
    print(f"During a merge step with cut-off {my_args.merge_cutoff}, size of rate matrix was reduced {size_pre_merge}->{size_post_merge}.",
          file=print_file)
    my_decomposer.cut_my_matrix(sh.get_magnitude_energy("Potential"), energy_limit=my_args.cut_cutoff)
    size_post_cut = my_decomposer.my_matrix.shape[0]
    print(f"During a cut step with energy limit {my_args.cut_cutoff}, size of rate matrix was reduced {size_post_merge}->{size_post_cut}.",
          file=print_file)

    # eigendecompoisition
    print(f"################## EIGENDECOMPOSITON OF THE RATE MATRIX ##################",
          file=print_file)
    eval, evec = my_decomposer.get_left_eigenvec_eigenval(*my_decomposer.get_default_settings_sqra())

    eigenval = eval.real
    eigenvec = evec.real

    print(f"Eigenvalues: {np.round(eigenval, 4)}",
          file=print_file)

    current_index_list = my_decomposer.current_index_list
    if not current_index_list:
        current_index_list = [[a] for a in range(len(eigenvec))]

    for i in range(5):
        num_extremes = 40
        magnitudes = eigenvec.T[i]
        most_positive = k_argmax_in_array(magnitudes, num_extremes)
        original_index_positive = []

        for mp in most_positive:
            original_index_positive.extend(current_index_list[mp])
        original_index_positive = np.array(original_index_positive)
        most_negative = k_argmax_in_array(-magnitudes, num_extremes)
        original_index_negative = []
        for mn in most_negative:
            original_index_negative.extend(current_index_list[mn])
        original_index_negative = np.array(original_index_negative)
        print(f"In {i}. eigenvector {num_extremes} most positive cells are "
              f"{list(original_index_positive + 1)}",
          file=print_file)
        print(f"and most negative {list(original_index_negative + 1)}.",
          file=print_file)

    t2 = time()
    my_log.update_after_calculation(t2-t1)

    print(f"Total time needed: ", end="",
          file=print_file)
    print(f"{timedelta(seconds=t2 - t1)} hours:minutes:seconds",
          file=print_file)
    my_log.add_information({"Final array size": size_post_cut, "Original array size": size_pre_merge})
    my_log.add_information({f"Eigenvalue {i}": eigenval[i] for i in range(6)})
    print_file.close()



if __name__ == '__main__':
    run_or_rerun()

