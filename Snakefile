from molgri.paths import PATH_OUTPUT_AUTOSAVE, PATH_OUTPUT_PT, PATH_OUTPUT_LOGGING, PATH_OUTPUT_ENERGIES
import numpy as np
from time import time, mktime
from datetime import timedelta
from datetime import datetime
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
        [f"{PATH_OUTPUT_AUTOSAVE}{grid_identifier}_full_array.npy" for grid_identifier in ALL_GRID_IDENTIFIERS],
        [f"{PATH_OUTPUT_PT}{pt_identifier}.gro" for pt_identifier in ALL_PT_IDENTIFIERS],
        [f"{PATH_OUTPUT_ENERGIES}{pt_identifier}.xvg" for pt_identifier in ALL_PT_IDENTIFIERS]

def log_the_run(name, input, output, log, params, time_used):
    logging.basicConfig(filename=log, level="INFO")
    logger = logging.getLogger(name)
    logger.info(f"SET UP: snakemake run with identifier {name}")
    logger.info(f"Input files: {input}")
    logger.info(f"Parameters: {params}")
    logger.info(f"Output files: {output}")
    logger.info(f"Log files: {log}")
    logger.info(f"Runtime of the total run: {timedelta(seconds=time_used)} hours:minutes:seconds")
    logger.info(f"This run was finished at: {datetime.fromtimestamp(time()).isoformat()}")

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
        np.save(output.full_array[0], fg.get_full_grid_as_array())
        # save geometric properties
        np.save(output.adjacency_array[0], fg.get_full_adjacency())
        np.save(output.borders_array[0],fg.get_full_borders())
        np.save(output.distances_array[0],fg.get_full_distances())
        np.save(output.volumes[0],fg.get_total_volumes())
        t2 = time()
        log_the_run(wildcards.grid_identifier, None, output, log[0], params, t2-t1)

rule run_pt:
    """
    This rule should produce the .gro and .xtc files of the pseudotrajectory.
    """
    input:
        m1=lambda wc: samples.loc[wc.pt_identifier, 'molecule1'],
        m2 = lambda wc: samples.loc[wc.pt_identifier,'molecule2'],
        grid = expand("{autosave}{grid_identifier}_full_array.npy", autosave=PATH_OUTPUT_AUTOSAVE,
            grid_identifier=lambda wc: samples.loc[wc.pt_identifier,"grid_identifier"], allow_missing=True)
    log: expand("{logpath}{pt_identifier}.log", logpath=PATH_OUTPUT_LOGGING, allow_missing=True)
    output:
        structure = expand("{ptpath}{pt_identifier}.gro",ptpath=PATH_OUTPUT_PT, allow_missing=True),
        trajectory = expand("{ptpath}{pt_identifier}.xtc", ptpath=PATH_OUTPUT_PT, allow_missing=True)
    #shell: "python -m molgri.scripts.generate_pt -m1 {input[0]} -m2 {input[1]} -b {params.b} -o {params.o} -t {params.t} -name {output} > {log}"
    run:
        t1 = time()
        from molgri.molecules.writers import PtWriter
        from molgri.molecules.pts import Pseudotrajectory
        from molgri.molecules.parsers import FileParser

        # load grid and molecules
        my_grid = np.load(input.grid[0])
        my_molecule1 = FileParser(input.m1).as_parsed_molecule()
        my_molecule2 = FileParser(input.m2).as_parsed_molecule()

        # create PT
        my_pt = Pseudotrajectory(my_molecule2, my_grid)

        # write out .gro and .xtc files
        my_writer = PtWriter("", my_molecule1)
        my_writer.write_full_pt(my_pt, path_structure=output.structure[0], path_trajectory=output.trajectory[0])
        t2=time()
        log_the_run(wildcards.pt_identifier,input,output,log[0],None,t2 - t1)


rule gromacs_rerun:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in 
    energies.
    """
    input:
        structure = expand("{ptpath}{pt_identifier}.gro",ptpath=PATH_OUTPUT_PT,allow_missing=True),
        trajectory = expand("{ptpath}{pt_identifier}.xtc",ptpath=PATH_OUTPUT_PT,allow_missing=True),
        topology = "../../../MASTER_THESIS/code/provided_data/topologies/H2O_H2O.top"
    log: expand("{logpath}{pt_identifier}_gromacs_rerun.log",logpath=PATH_OUTPUT_LOGGING,allow_missing=True)
    output: expand("{energypath}{pt_identifier}.xvg",energypath=PATH_OUTPUT_ENERGIES, allow_missing=True)
    # use with arguments like path_structure path_trajectory path_topology path_default_files path_output_energy
    shell: "molgri/scripts/gromacs_rerun_script.sh {wildcards.pt_identifier} {input.structure[0]} {input.trajectory[0]} {input.topology} {output} > {log}"
