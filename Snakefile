from molgri.paths import PATH_OUTPUT_AUTOSAVE, PATH_OUTPUT_PT, PATH_OUTPUT_LOGGING, PATH_OUTPUT_ENERGIES, PATH_OUTPUT_PLOTS
import numpy as np
from time import time, mktime
from datetime import timedelta
from datetime import datetime
import logging


# todo: create a report
report: "snakemake_workflow.rst"
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
        [f"{PATH_OUTPUT_AUTOSAVE}{pt_identifier}-{grid_identifier}_eigenvectors.npy" for pt_identifier in ALL_PT_IDENTIFIERS for grid_identifier in ALL_GRID_IDENTIFIERS],
        [f"{PATH_OUTPUT_PLOTS}{pt_identifier}-{grid_identifier}_eigenvalues.pdf" for pt_identifier in ALL_PT_IDENTIFIERS  for grid_identifier in ALL_GRID_IDENTIFIERS],
        [f"{PATH_OUTPUT_AUTOSAVE}{pt_identifier}-{grid_identifier}_vmdlog" for pt_identifier in ALL_PT_IDENTIFIERS for grid_identifier in ALL_GRID_IDENTIFIERS],


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
        adjacency_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_adjacency_array.npz",
        distances_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_distances_array.npz",
        borders_array= f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_borders_array.npz",
        volumes = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_volumes.npy",
    log: f"{PATH_OUTPUT_LOGGING}{{grid_identifier}}_full_array.log"
    params:
        b = lambda wc: grids.loc[wc.grid_identifier,"b_grid_name"],
        o = lambda wc: grids.loc[wc.grid_identifier,"o_grid_name"],
        t = lambda wc: grids.loc[wc.grid_identifier,"t_grid_name"]
    run:
        t1 = time()
        from molgri.space.fullgrid import FullGrid
        from scipy import sparse
        fg = FullGrid(params.b, params.o, params.t)

        # save full array
        np.save(output.full_array, fg.get_full_grid_as_array())
        # save geometric properties
        sparse.save_npz(output.adjacency_array, fg.get_full_adjacency())
        sparse.save_npz(output.borders_array,fg.get_full_borders())
        sparse.save_npz(output.distances_array,fg.get_full_distances())
        np.save(output.volumes,fg.get_total_volumes())
        t2 = time()
        log_the_run(wildcards.grid_identifier, None, output, log[0], params, t2-t1)


def get_run_pt_input(wc):
    m1 = samples.loc[wc.pt_identifier, 'molecule1']
    m2 = samples.loc[wc.pt_identifier, 'molecule2']
    # you should obtain the grid specified in the project table
    grid = f"{PATH_OUTPUT_AUTOSAVE}{wc.grid_identifier}_full_array.npy"
    return [m1, m2, grid]

rule run_pt:
    """
    This rule should produce the .gro and .xtc files of the pseudotrajectory.
    """
    input:
        get_run_pt_input
    log: f"{PATH_OUTPUT_LOGGING}{{pt_identifier}}-{{grid_identifier}}_pt.log"
    output:
        structure = f"{PATH_OUTPUT_PT}{{pt_identifier}}-{{grid_identifier}}.gro",
        trajectory = f"{PATH_OUTPUT_PT}{{pt_identifier}}-{{grid_identifier}}.xtc",
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
        structure = f"{PATH_OUTPUT_PT}{{pt_identifier}}-{{grid_identifier}}.gro",
        trajectory = f"{PATH_OUTPUT_PT}{{pt_identifier}}-{{grid_identifier}}.xtc",
        topology = "../../../MASTER_THESIS/code/provided_data/topologies/H2O_H2O.top"
    log: f"{PATH_OUTPUT_LOGGING}{{pt_identifier}}-{{grid_identifier}}_gromacs_rerun.log"
    output: f"{PATH_OUTPUT_ENERGIES}{{pt_identifier}}-{{grid_identifier}}_energy.xvg"
    # use with arguments like path_structure path_trajectory path_topology path_default_files path_output_energy
    shell: "molgri/scripts/gromacs_rerun_script.sh {wildcards.pt_identifier}-{wildcards.grid_identifier} {input.structure} {input.trajectory} {input.topology} {output} > {log}"


rule run_sqra:
    """
    As input we need: energies, adjacency, volume, borders, distances.
    As output we want to have the rate matrix.
    """
    input:
        energy = f"{PATH_OUTPUT_ENERGIES}{{pt_identifier}}-{{grid_identifier}}_energy.xvg",
        distances_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_distances_array.npz",
        borders_array= f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_borders_array.npz",
        volumes = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_volumes.npy",
    log: f"{PATH_OUTPUT_LOGGING}{{pt_identifier}}-{{grid_identifier}}_rate_matrix.log"
    output:
        rate_matrix = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_rate_matrix.npz",
        index_list = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_index_list.npy",
    params:
        D =1, # diffusion constant
        T=273,  # temperature in K
        energy_type = "Potential",
        upper_limit= 10,
        lower_limit= 0.1
    run:
        t1 = time()
        from molgri.molecules.parsers import XVGParser
        from molgri.molecules.rate_merger import (determine_rate_cells_with_too_high_energy, delete_rate_cells,
                                                  determine_rate_cells_to_join, merge_matrix_cells)
        from scipy.constants import k as kB, N_A
        from scipy.sparse import coo_array
        from scipy import sparse

        # load input files
        all_volumes = np.load(input.volumes)
        all_surfaces = sparse.load_npz(input.borders_array)
        all_distances = sparse.load_npz(input.distances_array)

        my_parsed = XVGParser(input.energy)
        energies = my_parsed.get_parsed_energy().get_energies(params.energy_type)

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

        # cut and merge
        rate_to_join = determine_rate_cells_to_join(all_distances, energies,
            bottom_treshold=params.lower_limit, T=params.T)
        transition_matrix, current_index_list = merge_matrix_cells(my_matrix=transition_matrix,
            all_to_join=rate_to_join,
            index_list=None)
        too_high = determine_rate_cells_with_too_high_energy(energies,energy_limit=params.upper_limit,T=params.T)
        transition_matrix, current_index_list = delete_rate_cells(transition_matrix,to_remove=too_high,
            index_list=current_index_list)
        # saving to file
        sparse.save_npz(output.rate_matrix, transition_matrix)
        np.save(output.index_list, np.array(current_index_list, dtype=object))
        t2 = time()
        log_the_run(wildcards.pt_identifier, input, output, log[0], params, t2-t1)

rule run_decomposition:
    """
    As output we want to have eigenvalues, eigenvectors. Es input we get a (sparse) rate matrix.
    """
    input: f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_rate_matrix.npz"
    log: f"{PATH_OUTPUT_LOGGING}{{pt_identifier}}-{{grid_identifier}}_eigendecomposition.log"
    output:
        eigenvalues = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_eigenvalues.npy",
        eigenvectors = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_eigenvectors.npy"
    params:
        sigma = 0,
        which = "LM",
        num_eigenvec = 6,
        tol = 1e-5,
        maxiter = 100000
    run:
        t1 = time()

        # loading
        from scipy.sparse.linalg import eigs
        from scipy import sparse
        my_matrix = sparse.load_npz(input[0])

        # calculation
        eigenval, eigenvec = eigs(my_matrix.T, params.num_eigenvec,tol=params.tol, maxiter=params.maxiter,
            which=params.which, sigma=params.sigma)
        if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
            eigenvec = eigenvec.real
            eigenval = eigenval.real
        # sort eigenvectors according to their eigenvalues
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]

        # saving to file
        np.save(output.eigenvalues, eigenval)
        np.save(output.eigenvectors,eigenvec)
        t2 = time()
        log_the_run(wildcards.pt_identifier, input, output, log[0], params, t2-t1)

rule run_eigenvalue_spectrum:
    """
    Make a plot of eigenvalues
    """
    input:
        eigenvalues = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_eigenvalues.npy"
    output:
        figure = f"{PATH_OUTPUT_PLOTS}{{pt_identifier}}-{{grid_identifier}}_eigenvalues.pdf"
    run:
        # next two lines seem stupid, but they are necessary when within snakemake
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from molgri.constants import DIM_SQUARE, DEFAULT_DPI
        eigenvals = np.load(input.eigenvalues)
        fig, ax = plt.subplots(1,1, figsize=DIM_SQUARE)
        xs = np.linspace(0, 1, num=len(eigenvals))
        ax.scatter(xs, eigenvals, s=5, c="black")
        for i, eigenw in enumerate(eigenvals):
            ax.vlines(xs[i], eigenw, 0, linewidth=0.5, color="black")
        ax.hlines(0, 0, 1, color="black")
        ax.set_ylabel(f"Eigenvalues")
        ax.axes.get_xaxis().set_visible(False)
        plt.savefig(output.figure, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close()


import re
def case_insensitive_search_and_replace(file_read, file_write, all_search_word, all_replace_word):
    with open(file_read, 'r') as file:
        file_contents = file.read()
        for search_word, replace_word in zip(all_search_word, all_replace_word):
            file_contents = file_contents.replace(search_word, replace_word)

    with open(file_write, 'w') as file:
        file.write(file_contents)


rule compile_vmd_log:
    """
    Input are the saved eigenvectors. Output = a vmd log that can be used later with:
    
    vmd <gro file> <xtc file>
    play <vmdlog file>
    """
    input:
        structure = f"{PATH_OUTPUT_PT}{{pt_identifier}}-{{grid_identifier}}.gro",
        trajectory = f"{PATH_OUTPUT_PT}{{pt_identifier}}-{{grid_identifier}}.xtc",
        eigenvectors = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_eigenvectors.npy",
        index_list = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_index_list.npy",
        # in the script only the numbers for frames need to be changed.
        script = "molgri/scripts/vmd_show_eigenvectors"
    output:
        vmdlog = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_vmdlog"
    params:
        num_extremes = 40,
        num_eigenvec = 4 # only show the first num_eigenvec
    run:
        from molgri.space.utils import k_argmax_in_array
        # load eigenvectors
        eigenvectors = np.load(input.eigenvectors)
        index_list = np.load(input.index_list, allow_pickle=True)

        # find the most populated states
        all_lists_to_insert = []
        for i, eigenvec in enumerate(eigenvectors.T[:params.num_eigenvec]):
            magnitudes = eigenvec
            # zeroth eigenvector only interested in max absolute values
            if i==0:
                # most populated 0th eigenvector
                most_populated = k_argmax_in_array(np.abs(eigenvec), params.num_extremes)
                original_index_populated = []
                for mp in most_populated:
                    original_index_populated.extend(index_list[mp])
                all_lists_to_insert.append(original_index_populated)
            else:
                most_positive = k_argmax_in_array(eigenvec, params.num_extremes)

                original_index_positive = []
                for mp in most_positive:
                    original_index_positive.extend(index_list[mp])
                original_index_positive = np.array(original_index_positive)
                most_negative = k_argmax_in_array(-magnitudes, params.num_extremes)
                original_index_negative = []
                for mn in most_negative:
                    original_index_negative.extend(index_list[mn])
                all_lists_to_insert.append(original_index_positive)
                all_lists_to_insert.append(original_index_negative)
        all_str_to_replace = [f"REPLACE{i}" for i in range(params.num_eigenvec*2-1)]
        all_str_to_insert = [', '.join(map(str,list(el))) for el in all_lists_to_insert]
        case_insensitive_search_and_replace(input.script,output.vmdlog,all_str_to_replace, all_str_to_insert)
        print("Your should now run:")
        print(f"vmd {input.structure} {input.trajectory}")
        print(f"play {output.vmdlog}")


# todo: combine pandas tables with all input tables, possibly parameters and runtimes
#import pandas as pd
#df_pt = pd.
#join(other, on=None
