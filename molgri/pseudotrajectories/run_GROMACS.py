import os
import subprocess
import shlex

from analysis.analyse_gromacs_run import generate_gro_results
from my_constants import *
from parsers.name_parser import NameParser


def run_one_calculation_GROMACS(mol1, mol2, grid_type, traj_type, N):
    cwd = os.getcwd()
    os.chdir(PATH_SCRIPTS)
    # H2O H2O ico 500 full
    subprocess.call(shlex.split(f'./rerun.sh {mol1} {mol2} {grid_type} {N} {traj_type}'))
    os.chdir(cwd)
    generate_gro_results(f"{mol1}_{mol2}_{grid_type}_{N}_{traj_type}")


def run_GROMACS_optimisation(mol1, mol2, traj_len):
    cwd = os.getcwd()
    os.chdir(PATH_SCRIPTS)
    # H2O H2O ico 500 full
    subprocess.call(shlex.split(f'./full_run.sh {mol1} {mol2} run {traj_len}'))
    os.chdir(cwd)
    generate_gro_results(f"{mol1}_{mol2}_run_{traj_len}")

if __name__ == "__main__":
    run_GROMACS_optimisation("protein0", "CL", int(1e5))