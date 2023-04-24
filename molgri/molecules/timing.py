import numpy as np
from time import time
import pandas as pd
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt

from molgri.space.translations import TranslationParser
from molgri.paths import PATH_OUTPUT_TIMING, PATH_OUTPUT_PLOTS
from molgri.constants import DIM_PORTRAIT, DEFAULT_DPI, DIM_LANDSCAPE

import timeit
import re
import subprocess as sp

def execute_scripts(script_setup):
    subprocess.call(script_setup)




def time_pt_creation(mol1, mol2, distances="[5, 10, 15]", N_rot_max=100, num_points=30):
    """
    Create PTs with the two provided molecules
    Args:
        mol1:
        mol2:
        distances:
        N_rot_max:

    Returns:

    """
    radii = TranslationParser(distances)
    num_samples = 5
    N_t = len(radii.get_trans_grid())
    N_rot_selections = np.linspace(1, N_rot_max, num=num_points, dtype=int)
    data = np.zeros((2, num_samples*len(N_rot_selections)))
    current_index = 0
    for N_rot in N_rot_selections:
        # perform molgri-pt
        commands = ["python", "-m", "molgri.scripts.generate_pt", "-m1", f"{mol1}", "-m2", f"{mol2}",
                        "-t", f"{distances}", "-o", f"{N_rot}", "-b", f"{N_rot}", "--recalculate"]
        t = timeit.Timer(f"execute_scripts({commands})", setup="from __main__ import execute_scripts")

        for _ in range(num_samples):
            # save data
            data[0][current_index] = N_rot * N_rot * N_t
            data[1][current_index] = t.timeit(1) / 60
            current_index += 1

    df = pd.DataFrame(data.T, columns=["Number of PT frames", "Generation time [min]"])
    print(df)
    df.to_csv(f"{PATH_OUTPUT_TIMING}{mol1}_{mol2}_time_{num_points}.csv")


def plot_timing(f_names):
    sns.set_context("talk")
    colors = ["red", "blue"]
    fig, ax = plt.subplots(1, 1, figsize=DIM_LANDSCAPE)
    for f_name, color in zip(f_names, colors):
        df = pd.read_csv(f_name)
        sns.lineplot(df, x="Number of PT frames", y="Generation time [min]", color=color, ax=ax)
    ax.set_xticks([100*100*3, 300*300*3, 500*500*3])
    xlabels = [f'{100*100*3}\n(100x100x3)', f'{300*300*3}\n(300x300x3)', f'{500*500*3}\n(500x500x3)']
    #xlabels_new = [re.sub("(.{8})", "\\1\n", label, 0, re.DOTALL) for label in xlabels]

    ax.set_xticklabels(xlabels)
    plt.tight_layout()
    plt.savefig(f"{PATH_OUTPUT_PLOTS}timing_pres.pdf", dpi=DEFAULT_DPI)
    #plt.show()

if __name__ == "__main__":
    num_points = 30
    #time_pt_creation("example_protein", "example_protein", N_rot_max=500, num_points=num_points)
    #time_pt_creation("example_pdb", "example_pdb", N_rot_max=500, num_points=num_points)
    #plot_timing(f_name=)
    #f"{PATH_OUTPUT_TIMING}example_pdb_example_pdb_time_{num_points}.csv",
    plot_timing([
                 f"{PATH_OUTPUT_TIMING}example_protein_example_protein_time_{num_points}.csv"])
