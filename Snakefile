configfile: "config.yaml"

import os
import matplotlib
from molgri.paths import *
matplotlib.use('pdf')  # necessary so matplotlib doesn't warn about threading
from molgri.grids import build_grid
from molgri.plotting import GridPlot



def get_bwa_map_input_fastqs(wildcards):
    print(os.getcwd())
    return "LICENSE"

rule generate_grid:
    output:
        expand("output/grid_files/{algorithm}_{N}.npy", algorithm="ico", N=15)
        #"molgri/output/grid_files/{algorithm}_{N}.npy"
    run:
        my_grid = build_grid(int(wildcards.N), wildcards.algorithm, use_saved=True, time_generation=True)
        my_grid.save_grid()

rule generate_grid_figures:
    input:
        "molgri/output/grid_files/{algorithm}_{N}.npy"
    output:
        "molgri/output/figures_grids/{algorithm}_{N}_grid.{ending}"
    threads: 1
    run:
        my_gp = GridPlot(f"{wildcards.algorithm}_{wildcards.N}",style_type=["talk", "empty"], fig_path=output[0])
        my_gp.create(save_ending=wildcards.ending) #
