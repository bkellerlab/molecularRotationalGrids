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
        [f"{PATH_OUTPUT_ENERGIES}{pt_identifier}.xvg" for pt_identifier in ALL_PT_IDENTIFIERS],
        #[f"{PATH_OUTPUT_AUTOSAVE}{pt_identifier}_{samples.loc[pt_identifier,'grid_identifier']}_rate_matrix.npy" for pt_identifier in ALL_PT_IDENTIFIERS]

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
        full_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_full_array.npy",
        adjacency_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_adjacency_array.npy",
        distances_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_distances_array.npy",
        borders_array= f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_borders_array.npy",
        volumes = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_volumes.npy",
    log: f"{PATH_OUTPUT_LOGGING}{{grid_identifier}}_full_array.log"
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
        log_the_run(wildcards.grid_identifier, None, output, log[0], params, t2-t1)


def get_run_pt_input(wc):
    m1 = samples.loc[wc.pt_identifier, 'molecule1']
    m2 = samples.loc[wc.pt_identifier, 'molecule2']
    # you should obtain the grid specified in the project table
    grid = f"{PATH_OUTPUT_AUTOSAVE}{samples.loc[wc.pt_identifier, 'grid_identifier']}_full_array.npy"
    return [m1, m2, grid]

rule run_pt:
    """
    This rule should produce the .gro and .xtc files of the pseudotrajectory.
    """
    input:
        get_run_pt_input
    log: f"{PATH_OUTPUT_LOGGING}{{pt_identifier}}_pt.log"
    output:
        structure = f"{PATH_OUTPUT_PT}{{pt_identifier}}.gro",
        trajectory = f"{PATH_OUTPUT_PT}{{pt_identifier}}.xtc",
    #shell: "python -m molgri.scripts.generate_pt -m1 {input[0]} -m2 {input[1]} -b {params.b} -o {params.o} -t {params.t} -name {output} > {log}"
    run:
        t1 = time()
        from molgri.molecules.writers import PtWriter
        from molgri.molecules.pts import Pseudotrajectory
        from molgri.molecules.parsers import FileParser

        # load grid and molecules
        my_grid = np.load(input[2])
        my_molecule1 = FileParser(input[0]).as_parsed_molecule()
        my_molecule2 = FileParser(input[1]).as_parsed_molecule()

        # create PT
        my_pt = Pseudotrajectory(my_molecule2, my_grid)

        # write out .gro and .xtc files
        my_writer = PtWriter("", my_molecule1)
        my_writer.write_full_pt(my_pt, path_structure=output.structure, path_trajectory=output.trajectory)
        t2=time()
        log_the_run(wildcards.pt_identifier,input,output,log[0],None,t2 - t1)


rule gromacs_rerun:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in 
    energies.
    """
    input:
        structure = f"{PATH_OUTPUT_PT}{{pt_identifier}}.gro",
        trajectory = f"{PATH_OUTPUT_PT}{{pt_identifier}}.xtc",
        topology = "../../../MASTER_THESIS/code/provided_data/topologies/H2O_H2O.top"
    log: f"{PATH_OUTPUT_LOGGING}{{pt_identifier}}_gromacs_rerun.log"
    output: f"{PATH_OUTPUT_ENERGIES}{{pt_identifier}}_energy.xvg"
    # use with arguments like path_structure path_trajectory path_topology path_default_files path_output_energy
    shell: "molgri/scripts/gromacs_rerun_script.sh {wildcards.pt_identifier} {input.structure} {input.trajectory} {input.topology} {output} > {log}"


rule run_sqra:
    """
    As input we need: energies, adjacency, volume, borders, distances.
    As output we want to have the rate matrix.
    """
    input:
        energy = f"{PATH_OUTPUT_ENERGIES}{{pt_identifier}}_energy.xvg",
        adjacency_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_adjacency_array.npy",
        distances_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_distances_array.npy",
        borders_array= f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_borders_array.npy",
        volumes = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_volumes.npy",
    log: f"{PATH_OUTPUT_LOGGING}{{pt_identifier}}_{{grid_identifier}}_rate_matrix.log"
    output: f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}_{{grid_identifier}}_rate_matrix.log"
    params:
        D =1, # diffusion constant
        T=273,  # temperature in K
        energy_type = "Potential"
    run:
        t1 = time()
        from molgri.molecules.parsers import XVGParser
        from scipy.constants import k as kB, N_A
        from scipy.sparse import coo_array
        # load input files
        all_volumes = np.load(input.volumes)
        all_surfaces = np.load(input.borders_array)
        all_distances = np.load(input.distances_array)
        energies = XVGParser(input.energy).get_parsed_energy().get_energies(params.energy_type)

        # calculating rate matrix
        # for sqra demand that each energy corresponds to exactly one cell
        assert len(energies) == len(all_volumes), f"{len(energies)} != {len(all_volumes)}"
        # you cannot multiply or divide directly in a coo format
        transition_matrix = params.D * all_surfaces  #/ all_distances
        transition_matrix = transition_matrix.tocoo()
        transition_matrix.data /= all_distances.tocoo().data
        # Divide every row of transition_matrix with the corresponding volume
        transition_matrix.data /= all_volumes[transition_matrix.row]
        # multiply with sqrt(pi_j/pi_i) = e**((V_i-V_j)*1000/(2*k_B*N_A*T))
        # gromacs uses kJ/mol as energy unit, boltzmann constant is J/K
        transition_matrix.data *= np.exp(np.round((energies[
                                                            transition_matrix.row] - energies[
                                                            transition_matrix.col]),14) * 1000 / (
                                                          2 * kB * N_A * params.T))
        # normalise rows
        sums = transition_matrix.sum(axis=1)
        sums = np.array(sums).squeeze()
        all_i = np.arange(len(all_volumes))
        diagonal_array = coo_array((-sums, (all_i, all_i)),shape=(len(all_i), len(all_i)))
        transition_matrix = transition_matrix.tocsr() + diagonal_array.tocsr()

        # saving to file
        np.save(output, transition_matrix)
        t2 = time()
        log_the_run(wildcards.pt_identifier, input, output, log[0], params, t2-t1)

# rule run_decomposition:
#     """
#     As output we want to have eigenvalues, eigenvectors
#     """
