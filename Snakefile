from molgri.paths import (PATH_OUTPUT_AUTOSAVE, PATH_OUTPUT_PT, PATH_OUTPUT_LOGGING, PATH_OUTPUT_ENERGIES,
                          PATH_OUTPUT_PLOTS, PATH_OUTPUT_TRAJECTORIES)
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
pepfile: "input/logbook/traj_pep.yaml"
traj_samples = pep.sample_table

ALL_GRID_IDENTIFIERS = list(grids.index)
ALL_PT_IDENTIFIERS = list(samples.index)
ALL_TRAJ_IDENTIFIERS = list(traj_samples.index)

TAUS = np.array([2, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150, 180, 200], dtype=int)

all_minimal_example = ([f"{PATH_OUTPUT_PLOTS}{pt_identifier}-{grid_identifier}_its_sqra.png" for pt_identifier in ALL_PT_IDENTIFIERS  for grid_identifier in ["small_ideal"]],
        [f"{PATH_OUTPUT_AUTOSAVE}{pt_identifier}-{grid_identifier}_vmdlog_sqra" for pt_identifier in ALL_PT_IDENTIFIERS for grid_identifier in ["small_ideal"]],)

all_actually_all = ([f"{PATH_OUTPUT_AUTOSAVE}{pt_identifier}-{grid_identifier}_eigenvectors.npy" for pt_identifier in ALL_PT_IDENTIFIERS for grid_identifier in ALL_GRID_IDENTIFIERS],
        [f"{PATH_OUTPUT_PLOTS}{pt_identifier}-{grid_identifier}_its_sqra.png" for pt_identifier in ALL_PT_IDENTIFIERS  for grid_identifier in ALL_GRID_IDENTIFIERS],
        [f"{PATH_OUTPUT_AUTOSAVE}{pt_identifier}-{grid_identifier}_vmdlog_sqra" for pt_identifier in ALL_PT_IDENTIFIERS for grid_identifier in ALL_GRID_IDENTIFIERS],
        [f"{PATH_OUTPUT_PLOTS}{traj_identifier}-{grid_identifier}_its_msm.png" for traj_identifier in ALL_TRAJ_IDENTIFIERS for grid_identifier in ALL_GRID_IDENTIFIERS],
        [f"{PATH_OUTPUT_AUTOSAVE}{traj_identifier}-{grid_identifier}_vmdlog_msm" for traj_identifier in ALL_TRAJ_IDENTIFIERS for grid_identifier in ALL_GRID_IDENTIFIERS],)

rule all:
    """Explanation: this rule is the first one, so it will be run. As an input, it should require the output files that 
    we get at the very end of our analysis because in this case all of the following rules that produce them must also 
    be called."""
    input:
        all_minimal_example


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

######################################################################################################################
#                                             GRIDS
######################################################################################################################

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


######################################################################################################################
#                                   PSEUDOTRAJECTORIES; SQRA
######################################################################################################################


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
        from molgri.molecules.transitions import SQRA
        from scipy import sparse

        # load input files
        all_volumes = np.load(input.volumes)
        all_surfaces = sparse.load_npz(input.borders_array)
        all_distances = sparse.load_npz(input.distances_array)

        my_parsed = XVGParser(input.energy)
        energies = my_parsed.get_parsed_energy().get_energies(params.energy_type)

        sqra = SQRA(energies=energies, volumes=all_volumes, distances=all_distances, surfaces=all_surfaces)
        rate_matrix = sqra.get_rate_matrix(params.D, params.T)
        rate_matrix, index_list = sqra.cut_and_merge(rate_matrix, T=params.T, lower_limit=params.lower_limit,
            upper_limit=params.upper_limit)

        # saving to file
        sparse.save_npz(output.rate_matrix, rate_matrix)
        np.save(output.index_list, np.array(index_list, dtype=object))
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
        from scipy import sparse
        from molgri.molecules.transitions import DecompositionTool

        # loading
        my_matrix = sparse.load_npz(input[0])

        # calculation
        dt = DecompositionTool(my_matrix,is_msm=False)
        all_eigenval, all_eigenvec = dt.get_decomposition(tol=params.tol,maxiter=params.maxiter,which=params.which,
            sigma=params.sigma)

        # saving to file
        np.save(output.eigenvalues,np.array(all_eigenval))
        np.save(output.eigenvectors,np.array(all_eigenvec))
        t2 = time()
        log_the_run(wildcards.pt_identifier, input, output, log[0], params, t2-t1)

rule run_plot_everything_sqra:
    """
    Make a plot of eigenvalues
    """
    input:
        eigenvalues = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_eigenvalues.npy",
        eigenvectors= f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_eigenvectors.npy"
    output:
        plot_eigenvectors=f"{PATH_OUTPUT_PLOTS}{{pt_identifier}}-{{grid_identifier}}_eigenvectors_sqra.png",
        plot_eigenvalues=f"{PATH_OUTPUT_PLOTS}{{pt_identifier}}-{{grid_identifier}}_eigenvalues_sqra.png",
        plot_its=f"{PATH_OUTPUT_PLOTS}{{pt_identifier}}-{{grid_identifier}}_its_sqra.png",
    run:
        t1 = time()
        from molgri.plotting.transition_plots import PlotlyTransitions

        pt = PlotlyTransitions(is_msm=False,path_eigenvalues=input.eigenvalues,path_eigenvectors=input.eigenvectors)
        # eigenvectors
        pt.plot_eigenvectors_flat()
        pt.save_to(output.plot_eigenvectors,height=1200)
        # eigenvalues
        pt.plot_eigenvalues()
        pt.save_to(output.plot_eigenvalues)
        # # its for msm
        pt.plot_its_as_line()
        pt.save_to(output.plot_its)
        # we could also plot the heatmap of the matrix, but it's honestly not that useful and can become very large
        t2 = time()
        log_the_run(wildcards.pt_identifier,input,output,None,None,t2 - t1)


rule compile_vmd_log:
    """
    Input are the saved eigenvectors. Output = a vmd log that can be used later with:
    
    vmd <gro file> <xtc file>
    play <vmdlog file>
    """
    input:
        eigenvectors = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_eigenvectors.npy",
        index_list = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_index_list.npy",
        # in the script only the numbers for frames need to be changed.
        script = "molgri/scripts/vmd_show_eigenvectors"
    output:
        vmdlog = f"{PATH_OUTPUT_AUTOSAVE}{{pt_identifier}}-{{grid_identifier}}_vmdlog_sqra"
    params:
        num_extremes = 40,
        num_eigenvec = 4 # only show the first num_eigenvec
    run:
        from molgri.plotting.create_vmdlog import show_eigenvectors
        # load eigenvectors
        eigenvectors = np.load(input.eigenvectors)
        index_list = np.load(input.index_list, allow_pickle=True)

        show_eigenvectors(input.script,output.vmdlog,eigenvector_array=eigenvectors,
            num_eigenvec=params.num_eigenvec,num_extremes=params.num_extremes, index_list=index_list)

######################################################################################################################
#                                       TRAJECTORIES; MSM
######################################################################################################################

def get_run_combine_molecules_input(wc):
    m1 = traj_samples.loc[wc.traj_identifier, 'molecule1']
    m2 = traj_samples.loc[wc.traj_identifier, 'molecule2']
    return [m1, m2]


rule run_combine_molecules:
    """
    Combine two files to create the .gro file needed for the gromacs trajectory run.
    """
    input:
        get_run_combine_molecules_input
    log: f"{PATH_OUTPUT_LOGGING}{{traj_identifier}}_combine.log"
    output:
        structure = f"{PATH_OUTPUT_TRAJECTORIES}{{traj_identifier}}.gro"
    params:
        start_dist = lambda wc: traj_samples.loc[wc.traj_identifier,"start_dist_A"]
    run:
        t1 = time()
        import MDAnalysis as mda
        from MDAnalysis import Merge

        central_molecule = mda.Universe(input[0])
        moving_molecule = mda.Universe(input[1])

        # center the both molecules
        com1 = central_molecule.atoms.center_of_mass()
        com2 = moving_molecule.atoms.center_of_mass()
        central_molecule.atoms.translate(-com1)
        moving_molecule.atoms.translate(-com2)
        # translate the second one
        moving_molecule.atoms.translate([0, 0, float(params.start_dist)])

        # merge and write
        merged_u = Merge(central_molecule.atoms, moving_molecule.atoms)
        with mda.Writer(output.structure) as writer:
            writer.write(merged_u)
        t2=time()
        log_the_run(wildcards.traj_identifier,input,output,log[0],None,t2 - t1)


rule gromacs_full_run:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in 
    energies.
    """
    input:
        structure = f"{PATH_OUTPUT_TRAJECTORIES}{{traj_identifier}}.gro",
        topology = "../../../MASTER_THESIS/code/provided_data/topologies/H2O_H2O.top"
    log: f"{PATH_OUTPUT_LOGGING}{{traj_identifier}}_gromacs_full_run.log"
    output:
        energy = f"{PATH_OUTPUT_ENERGIES}{{traj_identifier}}_energy.xvg",
        trajectory = f"{PATH_OUTPUT_TRAJECTORIES}{{traj_identifier}}.trr"
    params:
        traj_len = lambda wc: traj_samples.loc[wc.traj_identifier,"traj_len"]
    # use with arguments like full_name path_structure path_trajectory path_topology path_output_energy
    shell: "molgri/scripts/gromacs_full_run_script.sh {wildcards.traj_identifier} {input.structure} {input.topology} {output.energy} {output.trajectory} {params.traj_len} > {log}"

rule run_trajectory_assignment:
    """
    A step before MSM - assign every frame of the trajectory to the corresponding cell

    As input we need the trajectory, structure and full array of the grid we wanna assign to.

    As output we get a cell index for every frame of the trajectory.
    """

    input:
        full_array = f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_full_array.npy",
        trajectory = f"{PATH_OUTPUT_TRAJECTORIES}{{traj_identifier}}.trr",
        structure = f"{PATH_OUTPUT_TRAJECTORIES}{{traj_identifier}}.gro",
        reference_m2 = lambda wc: traj_samples.loc[wc.traj_identifier,"molecule2"]
    log: f"{PATH_OUTPUT_LOGGING}{{traj_identifier}}-{{grid_identifier}}_trajectory_assignment.log"
    output:
        assignments=f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_assignments.npy",
    run:
        from molgri.molecules.transitions import AssignmentTool

        # using all inputs
        my_grid = np.load(input.full_array)
        at = AssignmentTool(my_grid,input.structure,input.trajectory,input.reference_m2)

        # saving output
        np.save(output.assignments,at.get_full_assignments())

rule run_msm:
    """
    As input we need: assignments.

    As output we want to have the transition matrices for different taus.
    """
    input:
        full_array=f"{PATH_OUTPUT_AUTOSAVE}{{grid_identifier}}_full_array.npy",
        assignments=f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_assignments.npy"
    log: f"{PATH_OUTPUT_LOGGING}{{traj_identifier}}-{{grid_identifier}}_transition_matrix.log"
    output:
        transition_matrix=f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_transition_matrix.npy",
    params:
        taus=TAUS,
        noncorrelated_windows=False,
    run:
        from molgri.molecules.transitions import MSM

        # load data
        my_grid = np.load(input.full_array)
        my_assignments = np.load(input.assignments)
        num_cells = len(my_grid)

        my_msm = MSM(assigned_trajectory=my_assignments, total_num_cells=num_cells)
        my_transition_matrices = my_msm.get_all_tau_transition_matrices(
            noncorrelated_windows=params.noncorrelated_windows, taus=params.taus)

        # save the result
        np.save(output.transition_matrix, my_transition_matrices, allow_pickle=True)

rule run_decomposition_msm:
    """
    As output we want to have eigenvalues, eigenvectors. Es input we get a (sparse) rate matrix.
    """
    input: f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_transition_matrix.npy"
    log: f"{PATH_OUTPUT_LOGGING}{{traj_identifier}}-{{grid_identifier}}_eigendecomposition.log"
    output:
        eigenvalues = f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_eigenvalues_msm.npy",
        eigenvectors = f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_eigenvectors_msm.npy"
    params:
        sigma = None,
        which = "LR",
        num_eigenvec = 6,
        tol = 1e-7,
        maxiter = 100000
    run:
        t1 = time()
        from molgri.molecules.transitions import DecompositionTool

        # loading
        my_matrix = np.load(input[0], allow_pickle=True)

        # calculation
        dt = DecompositionTool(my_matrix, is_msm=True)
        all_eigenval, all_eigenvec = dt.get_decomposition(tol=params.tol, maxiter=params.maxiter, which=params.which,
            sigma=params.sigma)

        # saving to file
        np.save(output.eigenvalues, np.array(all_eigenval))
        np.save(output.eigenvectors, np.array(all_eigenvec))
        t2 = time()
        log_the_run(wildcards.traj_identifier, input, output, log[0], params, t2-t1)


rule run_plot_everything_msm:
    """
    Some stuff to plot after a MSM calculation: eigenvalues, ITS, eigenvectors
    """
    input:
        eigenvalues = f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_eigenvalues_msm.npy",
        eigenvectors = f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_eigenvectors_msm.npy"
    log: f"{PATH_OUTPUT_LOGGING}{{traj_identifier}}-{{grid_identifier}}_msm_plotting.log"
    params:
        taus = TAUS
    output:
        plot_eigenvectors = f"{PATH_OUTPUT_PLOTS}{{traj_identifier}}-{{grid_identifier}}_eigenvectors_msm.png",
        plot_eigenvalues = f"{PATH_OUTPUT_PLOTS}{{traj_identifier}}-{{grid_identifier}}_eigenvalues_msm.png",
        plot_its = f"{PATH_OUTPUT_PLOTS}{{traj_identifier}}-{{grid_identifier}}_its_msm.png",
    run:
        t1 = time()
        from molgri.plotting.transition_plots import PlotlyTransitions
        pt = PlotlyTransitions(is_msm=True, path_eigenvalues=input.eigenvalues, path_eigenvectors=input.eigenvectors,
            tau_array=params.taus)
        # eigenvectors
        pt.plot_eigenvectors_flat(index_tau=10)
        pt.save_to(output.plot_eigenvectors, height=1200)
        # eigenvalues
        pt.plot_eigenvalues(index_tau=10)
        pt.save_to(output.plot_eigenvalues)
        # # its for msm
        pt.plot_its_msm()
        pt.save_to(output.plot_its)
        # we could also plot the heatmap of the matrix, but it's honestly not that useful and can become very large
        t2 = time()
        log_the_run(wildcards.traj_identifier, input, output, log[0], params, t2-t1)

rule compile_vmd_log_msm:
    """
    Input are the saved eigenvectors. Output = a vmd log that can be used later with:

    vmd <gro file> <xtc file>
    play <vmdlog file>
    """
    input:
        eigenvectors=f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_eigenvectors_msm.npy",
        # in the script only the numbers for frames need to be changed.
        script="molgri/scripts/vmd_show_eigenvectors"
    output:
        vmdlog=f"{PATH_OUTPUT_AUTOSAVE}{{traj_identifier}}-{{grid_identifier}}_vmdlog_msm"
    params:
        num_extremes=40,
        num_eigenvec=4,  # only show the first num_eigenvec
        tau_index =10
    run:
        from molgri.plotting.create_vmdlog import show_eigenvectors

        # load eigenvectors
        eigenvectors = np.load(input.eigenvectors)

        show_eigenvectors(input.script, output.vmdlog, eigenvector_array=eigenvectors[params.tau_index],
            num_eigenvec=params.num_eigenvec, num_extremes=params.num_extremes)


# todo: combine pandas tables with all input tables, possibly parameters and runtimes
#import pandas as pd
#df_pt = pd.
#join(other, on=None
