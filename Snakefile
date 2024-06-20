from molgri.paths import PATH_OUTPUT_AUTOSAVE, PATH_OUTPUT_PT, PATH_OUTPUT_LOGGING
import numpy as np
from time import time
from datetime import timedelta
import logging

pepfile: "input/logbook/grid_pep.yaml"
grids = pep.sample_table
pepfile: "input/logbook/pt_pep.yaml"
samples = pep.sample_table

ALL_GRID_IDENTIFIERS = list(grids.index)
ALL_PT_IDENTIFIERS = list(samples.index)

rule all:
    """Explanation: this rule is the first one, so it will be run. As an input, it should require the output files that 
    we get at the very end of our analysis because in this case all of the following rules that produce them must also 
    be called."""
    input:
        [f"{PATH_OUTPUT_AUTOSAVE}{grid_identifier}_full_array.npy" for grid_identifier in ALL_GRID_IDENTIFIERS]
        #[f"{PATH_OUTPUT_PT}{pt_identifier}.gro" for pt_identifier in ALL_PT_IDENTIFIERS],

def log_the_run(name, input, output, log, params, time_used):
    logging.basicConfig(filename=log[0], level="INFO")
    logger = logging.getLogger(name)
    logger.info(f"SET UP: snakemake run with identifier {name}")
    logger.info(f"Input files: {input}")
    logger.info(f"Parameters: {params}")
    logger.info(f"Output files: {output}")
    logger.info(f"Log files: {log}")
    logger.info(f"Runtime of the total run: {timedelta(seconds=time_used)} hours:minutes:seconds")
    logger.info(f"This run was finished at: {time()}")

rule run_grid:
    """
    This rule should provide a full grid and its geometric parameters.
    """
    output:
        full_array = expand("{autosave}{grid_identifier}_full_array.npy", autosave=PATH_OUTPUT_AUTOSAVE, allow_missing=True),
        adjacency_array = expand("{autosave}{grid_identifier}_adjacency_array.npy", autosave=PATH_OUTPUT_AUTOSAVE, allow_missing=True),
        distances_array= expand("{autosave}{grid_identifier}_distances_array.npy", autosave=PATH_OUTPUT_AUTOSAVE, allow_missing=True),
        borders_array= expand("{autosave}{grid_identifier}_borders_array.npy", autosave=PATH_OUTPUT_AUTOSAVE, allow_missing=True),
        volumes = expand("{autosave}{grid_identifier}_volumes.npy", autosave=PATH_OUTPUT_AUTOSAVE, allow_missing=True)
    log: expand("{logpath}{grid_identifier}.log", logpath=PATH_OUTPUT_LOGGING, allow_missing=True)
    params:
        b = lambda wc: grids.loc[wc.grid_identifier,"b_grid_name"],
        o = lambda wc: grids.loc[wc.grid_identifier,"o_grid_name"],
        t = lambda wc: grids.loc[wc.grid_identifier,"t_grid_name"]
    run:
        t1 = time()
        from molgri.space.fullgrid import FullGrid
        fg = FullGrid(params.b, params.o, params.t)

        # save full array
        np.save(output.full_array, fg.get_full_grid_as_array())
        # save geometric properties
        np.save(output.adjacency_array, fg.get_full_adjacency())
        np.save(output.borders_array,fg.get_full_borders())
        np.save(output.distances_array,fg.get_full_distances())
        np.save(output.volumes,fg.get_total_volumes())
        t2 = time()
        log_the_run(wildcards.grid_identifier, None, output, log, params, t2-t1)


# def determine_pt_input_files(wc):
#     m1 = lambda wc: samples.loc[wc.pt_identifier, 'molecule1']
#     print(m1)
#     m2 = lambda wc: samples.loc[wc.pt_identifier, 'molecule2']
#     b = lambda wc: samples.loc[wc.pt_identifier, "b_grid_name"]
#     o = lambda wc: samples.loc[wc.pt_identifier, "o_grid_name"]
#     t = lambda wc: samples.loc[wc.pt_identifier, "t_grid_name"]
#     return m1, m2, f"{b}_{o}_{t}"
#
# rule run_pt:
#     """
#     This rule should produce the .gro and .xtc files of the pseudotrajectory.
#     """
#     input:
#         unpack(determine_pt_input_files)
#     log: "output/data/logging/{pt_identifier}.log"
#     output:
#         structure = "output/data/pt_files/{pt_identifier}.gro",
#         trajectory = "output/data/pt_files/{pt_identifier}.xtc"
#     params:
#         b = lambda wc: samples.loc[wc.pt_identifier,"b_grid_name"],
#         o = lambda wc: samples.loc[wc.pt_identifier,"o_grid_name"],
#         t = lambda wc: samples.loc[wc.pt_identifier,"t_grid_name"]
#     shell: "python -m molgri.scripts.generate_pt -m1 {input[0]} -m2 {input[1]} -b {params.b} -o {params.o} -t {params.t} -name {output} > {log}"
    # run:
    #     from molgri.molecules.writers import PtWriter
    #     from molgri.molecules.pts import Pseudotrajectory
    #
    #     # timestamp and execution time
    #
    #     # load grid and molecules
    #     my_grid = np.load()
    #     my_molecule2 =
    #
    #     # create PT
    #     my_pt = Pseudotrajectory(my_molecule2, my_grid)
    #
    #     # write out .gro and .xtc files
    #     my_writer = PtWriter("",)
    #     my_writer.write_full_pt(path_structure=output.structure, path_trajectory=output.trajectory)



