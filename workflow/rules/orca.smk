import os.path

import numpy as np
import pandas as pd

from workflow.helpers.io import get_num_atoms, read_object, write_object

from workflow.helpers.orca_reader import AVOGADRO_CONSTANT, HARTREE_TO_J

PATH_REMOTE = "../<pseudosimulation>"

N_structures_per_batch = int(config["curta"]["num_structures_per_batch"])
N_batches = int(config["curta"]["num_batches"])


NUM_GRID_POINTS_CURTA = N_structures_per_batch * N_batches

########################   HERE THE OPTION OF COMLETELY SPLIT TRAJECTORIES (BATCH + FRAME) ################

def _determine_batch_subfolders():
    all_paths = []
    for batch in range(N_batches):
        section_size = NUM_GRID_POINTS_CURTA // N_batches
        batch_start_index = batch*section_size
        batch_end_index = np.min([(batch+1) * section_size, NUM_GRID_POINTS_CURTA])
        all_paths.extend([f"batch_{batch}/{str(i).zfill(10)}/" for i in range(batch_start_index, batch_end_index)])

    return all_paths

def _determine_batch_folders(wildcards, file_needed):
    all_paths = []
    for batch in range(N_batches):
        section_size = NUM_GRID_POINTS_CURTA // N_batches
        batch_start_index = batch*section_size
        batch_end_index = np.min([(batch+1) * section_size, NUM_GRID_POINTS_CURTA])
        all_paths.extend([f"{wildcards.where}batch_{batch}/{str(i).zfill(10)}/{file_needed}" for i in range(batch_start_index, batch_end_index)])
    return all_paths

def determine_output_files_in_batches(wildcards):
    return _determine_batch_folders(wildcards, "orca.out")

def determine_input_files_in_batches(wildcards):
    return _determine_batch_folders(wildcards,"orca.inp")

paths = [f"{PATH_REMOTE}{my_path}orca.out" for my_path in
                            _determine_batch_subfolders() if
                            os.path.exists(f"{PATH_REMOTE}{my_path}orca.out")]

rule completely_split_trajectory:
    """
    If needed (e.g. for orca calculations) split a single file trajectory info a folder of single-structure files named
    from 0000000000 to the total num of points.

    Warning, this is not a general tool for any .xyz files but specifically for my pseudotrajectory file.
    """
    input:
        trajectory="<pseudosimulation>trajectory.<ext_trj>"
    output:
        all_out = expand(f"{PATH_REMOTE}{{specific_paths}}trajectory.<ext_trj>",
            specific_paths=_determine_batch_subfolders(), allow_missing=True),
        to_touch = touch("<pseudosimulation>trajectory_completely_split.touch")
    run:
        with open(input.trajectory, "r") as f:
            all_lines = f.readlines() # throwing away the last \n line

        split_len = len(all_lines) // NUM_GRID_POINTS_CURTA
        print("total len ", len(all_lines))

        for i, output_file in enumerate(output.all_out):
            with open(output_file, "w") as f:
                f.writelines(all_lines[i*split_len:(i+1)*split_len])

rule provide_all_inps:
    input:
        expand(f"{PATH_REMOTE}{{specific_paths}}orca.inp",
            specific_paths=_determine_batch_subfolders())
    output:


rule inp_into_every_batch_every_subfolder:
    """
    We want to create subdirectories with chunks of the trajectory.
    """
    input:
        inp = "<inputs_orca>orca.inp"
    params:
        N_structures_per_chunk=N_structures_per_batch
    output:
        inp = expand(f"{PATH_REMOTE}{{specific_paths}}orca.inp",
            specific_paths=_determine_batch_subfolders()),
        touchfile = touch("<pseudosimulation>all_orca_inp_exist.touch")
    run:
        for el in output.inp:
            shell("""
            cp  {input.inp} {el}
            """)

rule copy_runfiles_to_curta:
    input:
        script_run_orca_job = f"<inputs_orca>run_ORCA.sh",
        script_submit_all_jobs = f"<inputs_orca>submit_on_curta.sh",
    output:
        script_run_orca_job=f"{PATH_REMOTE}run_ORCA.sh",
        script_submit_all_jobs=f"{PATH_REMOTE}submit_on_curta.sh",
    shell:
        """
        cp  {input.script_run_orca_job} {output.script_run_orca_job}
        cp  {input.script_submit_all_jobs} {output.script_submit_all_jobs}
        """


rule energies_from_curta_to_local:
    input:
        remote_csv_file = f"{PATH_REMOTE}orca_data.csv",
        #report_failed = f"{PATH_REMOTE}report_failed.txt"
    output:
        csv_file = f"<pseudosimulation>energy.csv"
    run:
        rough_csv = read_object(input.remote_csv_file)
        rough_csv["SP Energy [Hartree]"] = pd.to_numeric(rough_csv["Final Single Point Energy"].map(lambda x: x.split()[-1] if isinstance(x, str) else x), errors="coerce")
        rough_csv["Dispersion [Hartree]"] = pd.to_numeric(rough_csv["Last Dispersion Correction"].map(lambda x: x.split()[-1] if isinstance(x,str) else x),errors="coerce")
        rough_csv["Energy [Hartree]"] = (rough_csv["SP Energy [Hartree]"] + rough_csv["Dispersion [Hartree]"])
        rough_csv["Energy [kJ/mol]"] = (rough_csv["SP Energy [Hartree]"] + rough_csv["Dispersion [Hartree]"] ) * HARTREE_TO_J * AVOGADRO_CONSTANT / 1000.0
        rough_csv["Total index"] = rough_csv.index.map(lambda x: int(x.split("/")[-2]))
        rough_csv["File"] = rough_csv.index
        rough_csv["Time [h:m:s]"]=pd.to_timedelta(rough_csv["Total Run Time"].map(lambda x: x.split(":")[-1].replace("msec", "ms") if isinstance(x, str) else x), errors="coerce")
        rough_csv["Time [s]"] = rough_csv["Time [h:m:s]"].dt.total_seconds()
        print("Length of rough csv ", len(rough_csv))
        clean_csv = rough_csv[["File", "Total index", "Energy [Hartree]", "Final Single Point Energy", "Energy [kJ/mol]", "Time [h:m:s]", "Time [s]"]]
        clean_csv = clean_csv.set_index("Total index").sort_index()
        print("Length of clean csv ", len(clean_csv))

        # failed_calcs = np.loadtxt(input.report_failed, dtype=str)
        # print(failed_calcs)
        # failed_calcs = [int(calc.split("/")[-1]) for calc in failed_calcs]
        # clean_csv.loc[clean_csv.index[failed_calcs], "Energy [kJ/mol]"] = np.nan

        write_object(clean_csv, output.csv_file)


########################   END: HERE THE OPTION OF COMPLETELY SPLIT TRAJECTORIES (BATCH + FRAME)  ################


rule trajectory_slice:
    """
    We want to extract just the frame with index frame_i from a full trajectory.
    """
    input:
        structure="<pseudosimulation>structure.<ext_str>",
        trajectory="<pseudosimulation>trajectory.<ext_trj>",
    output:
        frame_gro="<pseudosimulation>trajectory_slices/frame_{frame_i}.<ext_str>",
    run:
        N_atoms = get_num_atoms(input.structure)
        block_len = N_atoms + 2
        frame_index = int(wildcards.frame_i)

        # which lines in the file to read
        start=frame_index*block_len +1
        end=(frame_index+1)*block_len

        shell(f"""sed -s -n {start},{end}p {input.trajectory} > {output.frame_gro}""")

rule trajectory_into_chunks:
    """
    We want to create subdirectories with chunks of the trajectory.
    """
    input:
        structure="<pseudosimulation>structure.<ext_str>",
        trajectory="<pseudosimulation>trajectory.<ext_trj>",
        grid_info="<outputs_network>grid_info.yaml"
    params:
        N_structures_per_chunk=N_structures_per_batch
    output:
        trajectories = expand(f"{PATH_REMOTE}batch_{{i}}/trajectory.xyz", i=range(N_batches))
    run:
        N_atoms = get_num_atoms(input.structure)
        block_len = N_atoms + 2
        frames_per_chunk = int(params.N_structures_per_chunk)

        total_N_frames = read_object(input.grid_info)["N_total"]
        total_N_directories = total_N_frames//frames_per_chunk

        # create the directories
        for trj in output.trajectories:
            Path(os.path.dirname(trj)).mkdir(parents=True,exist_ok=True)
        # create the split files

        start_of_chunk = list(range(1, total_N_frames*block_len+1, frames_per_chunk*block_len))

        for i, start_line_number in enumerate(start_of_chunk):
            end_line_number = start_line_number + frames_per_chunk*block_len -1
            shell(f"""sed -s -n {start_line_number},{end_line_number}p {input.trajectory} > {output.trajectories[i]}""")


rule inp_into_every_batch:
    """
    We want to create subdirectories with chunks of the trajectory.
    """
    input:
        inp = f"{PATH_REMOTE}orca.inp"
    params:
        N_structures_per_chunk=N_structures_per_batch
    output:
        inp = expand(f"{PATH_REMOTE}batch_{{i}}/orca.inp", i=range(N_batches))
    run:
        from shutil import copy

        for out in output.inp:
            copy(input.inp, out)


# rule extract_energies_on_curta:
#     input:
#         orca_output = expand(f"{PATH_REMOTE}batch_{{i}}/orca.out", i=range(10))
#     output:
#         csv_file = f"{PATH_REMOTE}orca.csv"
#     params:
#         N_structures_per_chunk=N_structures_per_batch
#     run:
#         from workflow.helpers.orca_reader import read_important_stuff_into_csv
#
#         read_important_stuff_into_csv(input.orca_output, output.csv_file, chunksize=int(params.N_structures_per_chunk))

# rule copy_energy_to_local:
#     """
#     For the pseudotrajectory, read the energy of each frame.
#     """
#     input:
#         energy=f"{PATH_REMOTE}orca.csv"
#     output:
#         energy_csv = "<pseudosimulation>energy.csv"
#     shell:
#         """
#         cp {input.energy} {output.energy_csv}
#         """

