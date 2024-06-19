"""
User script for generating  SQRA matrices and calculating their eigenvectors/eigenvalues.
"""

import argparse
from time import time
from datetime import timedelta

import warnings

from build.lib.molgri.scripts.set_up_io import freshly_create_all_folders
from molgri.scripts.abstract_scripts import ScriptLogbook

warnings.filterwarnings("ignore")

# TODO: define total_N and generate in all dimensions uniform grid?

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-a', type=str,  required=True,
                           help='first required argument')
requiredNamed.add_argument('-b', type=int, required=True, help='second required argument')

parser.add_argument('--option1', type=str, help='first optional argument')
parser.add_argument('--option2', type=str, help='second optional argument')

parser.add_argument('--recalculate', action='store_true',
                    help='recalculate the grid even if a saved version already exists')

def run_or_rerun(to_run):
    freshly_create_all_folders()
    my_args = parser.parse_args()

    use_saved = not my_args.recalculate

    my_log = ScriptLogbook(parser)
    print("MY LOG", my_log.my_index, my_log.is_newly_assigned)
    name_print_file = f"{PATH_OUTPUT_AUTOSAVE}{my_log.class_name}_{my_log.my_index}.txt"

    # IF use_saved=True AND you find the relevant file, do not re-create anything, print the saved output message
    if use_saved and not my_log.is_newly_assigned:
        print("PLEASE NOTE: This exact calculation has been done before. These are old saved results.", end=" ")
        print("If you do need a fresh calculation, use the option --recalculate")
    else:
        with open(name_print_file, "w") as f:
            to_run(my_args, my_log, use_saved, f)

    with open(name_print_file, mode="r") as f:
        print(f.read())



def run_specific_script(my_args, my_log, use_saved, print_file):
    t1 = time()
    print(f"During a merge step with cut-off {my_args.merge_cutoff}, size of rate matrix was reduced {size_pre_merge}->{size_post_merge}.",
          file=print_file)
    t2 = time()
    my_log.update_after_calculation(t2-t1)

    print(f"Total time needed: ", end="",
          file=print_file)
    print(f"{timedelta(seconds=t2 - t1)} hours:minutes:seconds",
          file=print_file)
    my_log.add_information({"Final array size": size_post_cut, "Original array size": size_pre_merge})


if __name__ == '__main__':
    run_or_rerun(run_specific_script)