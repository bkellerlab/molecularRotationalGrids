import os
import matplotlib
matplotlib.use('pdf')
from molgri.grids import build_grid
from molgri.plotting import GridPlot

ALGORITHMS = ["ico", "cube3D", "cube4D", "randomQ", "randomE", "systemE"]

def get_bwa_map_input_fastqs(wildcards):
    print(os.getcwd())
    return "LICENSE"

rule generate_grid:
    output:
        "molgri/output/grid_files/{algorithm}_{N}.npy"
    run:
        my_grid = build_grid(wildcards.algorithm, int(wildcards.N), use_saved=True, time_generation=True)
        my_grid.save_grid(output[0])

rule generate_grid_figures:
    output:
        "molgri/output/figures_grids/{algorithm}_{N}_grid.{ending}"
    threads: 1
    run:
        my_gp = GridPlot(f"{wildcards.algorithm}_{wildcards.N}",style_type=["talk", "empty"], fig_path=output[0])
        my_gp.create(save_ending=wildcards.ending) #
