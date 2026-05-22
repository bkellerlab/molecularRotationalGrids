"""
In these rules we use gromacs to quickly and efficiently manipulate trajectory objects: eg. write out specific frames
for plotting, center and align to reference, extract only the second molecule or the center of mass etc.
"""
from MDAnalysis import Merge

from workflow.helpers.io import read_object, from_xvg_to_csv_energy

rule copy_other_gromacs_input:
    """
    Copy the rest of necessary files to start a GROMACS calculation.
    """
    input:
        dimer_topology = f"<inputs_gromacs>topol.top",
        select_energy = f"<inputs_gromacs>select_energy",
        index = f"<inputs_gromacs>index.ndx",
        force_field_stuff = f"<inputs_gromacs>force_field_stuff/"
    output:
        dimer_topology = f"<pseudosimulation>topol.top",
        select_energy = f"<pseudosimulation>select_energy",
        index = f"<pseudosimulation>index.ndx",
        force_field_stuff = directory(f"<pseudosimulation>force_field_stuff/")
    run:
        import shutil
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.dimer_topology, output.dimer_topology)
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.index,output.index)
        shutil.copytree(input.force_field_stuff,output.force_field_stuff, dirs_exist_ok=True)

rule copy_mdp_files:
    """
    Copy only mdp files that are required - e.g. for rerun you do not need a minim.mdp and nvt.mdp.
    """
    input:
        mdp = f"<inputs_gromacs>{{file_name}}.mdp",
    output:
        mdp = f"<pseudosimulation>{{file_name}}.mdp",
    run:
        import shutil
        shutil.copy(input.mdp,output.mdp)

rule add_timesteps_to_pt:
    """
    Pseudotrajectories obviously don't have timesteps as they are not time-dependent, but we need to add fake timstamps
    to use gromacs slicing options since they refer to time rather than to frame index.
    """
    input:
        trajectory=f"<pseudosimulation>trajectory.xtc",
        structure_tpr=f"<pseudosimulation>structure.gro",
        index=f"<pseudosimulation>index.ndx",
        runfile=f"<pseudosimulation>production.mdp"
    output:
        trajectory=f"<pseudosimulation>trajectory_with_timesteps.xtc",
    run:
        from workflow.helpers.io import read_from_mdrun

        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))
        timesteps = writeout * time_step_ps
        shell("""
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "0\n" |  gmx22 trjconv -f {input.trajectory} -s  {input.structure_tpr} -o {output.trajectory} -n {input.index} -timestep {timesteps}

        """)


def input_function_trajectory_slice(wc):
    """
    We use trajectory slices (.gro files containing exactly one frame) a lot in plotting of structures found at
    particular indices.
    """
    if "assignment" in wc.path:
        trajectory_name = "wrapped_trajectory"
        path_other_files = "<simulation>"
    elif "pseudosimulation" in wc.path:
        trajectory_name = "trajectory_with_timesteps"
        path_other_files = "<pseudosimulation>"
    else:
        trajectory_name = "trajectory"
        path_other_files = "<simulation>"
    return {"trajectory": f"{wc.path}{trajectory_name}.xtc",
            "structure_tpr": f"{path_other_files}structure.gro",
            "index": f"{path_other_files}index.ndx",
            "runfile": f"{path_other_files}production.mdp"}


rule trajectory_slice:
    """
    We want to extract just the frame with index frame_i from a full trajectory.
    """
    input:
        unpack(input_function_trajectory_slice)
    shadow: "minimal"
    output:
        frame_gro="{path}trajectory_slices/frame_{frame_i}.gro",
    run:
        from workflow.helpers.io import read_from_mdrun
        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))
        selected_time = int(wildcards.frame_i) * writeout * time_step_ps
        shell("""
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "0\n" |  gmx22 trjconv -f {input.trajectory} -s  {input.structure_tpr} -o {output.frame_gro} -n {input.index} -dump {selected_time}
        """)

rule shortened_trajectory:
    """
    This is helpful for testing new analysis methods without waiting forever for results. Just use shortened_trajectory
    instead of trajectory in input.
    """
    input:
        unpack(input_function_trajectory_slice)
    benchmark:
        repeat("{path}duration_gromacs_shortening.txt",1)
    shadow: "minimal"
    params:
        length_shortened_trajectory_ps = int(1000 * float(config["analysis"]["length_shortened_trajectory_ns"]))
    output:
        trajectory="{path}shortened_trajectory.xtc",
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "0\n" |  gmx22 trjconv -f {input.trajectory} -s  {input.structure_tpr} -o {output.trajectory} -n {input.index} -e {params.length_shortened_trajectory_ps}
        """

rule copy_simulation_to_wrapped:
    input:
        energy = "<simulation>energy.xvg",
        structure = "<simulation>structure.gro",
    output:
        energy = "<outputs_assignment>energy.xvg",
        structure = "<outputs_assignment>structure.gro",
    shell:
        """
        cp {input.energy} {output.energy}
        cp {input.structure} {output.structure}
        """

checkpoint create_energy_csv_trajectory:
    """
    For the pseudotrajectory, read the energy of each frame.
    """
    input:
        energy="{path}energy.xvg",
    output:
        energy_csv = "{path}energy.csv"
    run:
        from_xvg_to_csv_energy(input.energy,output.energy_csv,config["analysis"]["energy_types"])

# rule gromacs_equilibration:
#     """
#     This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
#     energies.
#     """
#     input:
#         structure=f"<outputs_gromacs>structure.gro",
#         runfile_minim=f"<outputs_gromacs>minim.mdp",
#         runfile_nvt=f"<outputs_gromacs>nvt.mdp",
#         index=f"<outputs_gromacs>index.ndx",
#         topology=f"<outputs_gromacs>topol.top",
#         force_field_stuff=f"<outputs_gromacs>force_field_stuff/"
#     shadow: "minimal"
#     output:
#         energy=f"<outputs_gromacs>nvt.gro",
#     shell:
#         """
#         initial_dir=$(pwd)
#         cd $(dirname {input.runfile_minim})
#         echo $(pwd)
#         export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
#         gmx22 grompp -f $(basename {input.runfile_minim}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o em.tpr -r $(basename {input.structure})
#         gmx22 mdrun -v -deffnm em
#
#         gmx22 grompp -f $(basename {input.runfile_nvt}) -c em.gro -r em.gro -p $(basename {input.topology}) -o nvt.tpr -n $(basename {input.index})
#         gmx22 mdrun -v -deffnm nvt
#
#         cd "$initial_dir" || exit
#         """
#
#
# rule gromacs_production:
#     """
#     This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
#     energies.
#     """
#     input:
#         structure=f"<outputs_gromacs>nvt.gro",
#         runfile=f"<outputs_gromacs>production.mdp",
#         topology=f"<outputs_gromacs>topol.top",
#         select_energy=f"<outputs_gromacs>select_energy",
#         force_field_stuff=f"<outputs_gromacs>force_field_stuff/",
#         index=f"<outputs_gromacs>index.ndx",
#     log:
#         log=f"<outputs_gromacs>logging_gromacs.log"
#     benchmark:
#         repeat(f"<outputs_gromacs>gromacs_benchmark.txt",1)
#     shadow: "minimal"
#     output:
#         structure_tpr=f"<outputs_gromacs>structure.tpr",
#         energy=f"<outputs_gromacs>energy.xvg",
#         original_trajectory=f"<outputs_gromacs>raw_trajectory.xtc"
#     shell:
#         """
#         initial_dir=$(pwd)
#         cd $(dirname {input.runfile})
#         export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
#         gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -r $(basename {input.structure}) -p $(basename {input.topology}) -o $(basename {output.structure_tpr}) -n $(basename {input.index})
#         gmx22 mdrun -v -deffnm structure -g $(basename {log.log})
#         mv structure.xtc $(basename {output.original_trajectory})
#         gmx22 energy -f structure.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
#         cd "$initial_dir" || exit
#         """

rule create_structure:
    input:
        molecule_1 = f"<pseudosimulation>molecule1.<ext_inp>",
        molecule_2 = f"<pseudosimulation>molecule2.<ext_inp>",
    output:
        structure = f"<pseudosimulation>structure.<ext_str>",
    run:
        from molgri.molecules.bimolecular import get_bimolecular_structure
        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)
        z_distance = float(config["grid"]["translation_subgrids_A"][-1][1])
        structure = get_bimolecular_structure(m1, m2, z_distance=z_distance)
        write_object(structure, output.structure)

rule postprocess_gromacs:
    input:
        original_trajectory = f"<simulation>raw_trajectory.xtc",
        structure_tpr=f"<simulation>structure.tpr",
        structure_gro=f"<pseudosimulation>structure.gro",
        index=f"<simulation>index.ndx",
    benchmark:
        repeat(f"<simulation>duration_gromacs_postprocessing.txt",1)
    shadow: "minimal"
    output:
        centered_trajectory = f"<simulation>centered_trajectory.xtc",
        trajectory=f"<simulation>trajectory.xtc",
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        # now fit to first frame
        echo "0\n" |  gmx22 trjconv -f {input.original_trajectory} -s {input.structure_tpr} -pbc mol -boxcenter tric -o {output.centered_trajectory} -n {input.index}
        echo "4\n0\n" |  gmx22 trjconv -fit trans -f {output.centered_trajectory} -o {output.trajectory} -s  {input.structure_gro} -n {input.index}
        """

rule gromacs_rerun:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure = f"<pseudosimulation>structure.gro",
        trajectory = f"<pseudosimulation>trajectory.xtc",
        runfile = f"<pseudosimulation>production.mdp",
        topology = f"<pseudosimulation>topol.top",
        index=f"<pseudosimulation>index.ndx",
        select_energy = f"<pseudosimulation>select_energy",
        force_field_stuff = f"<pseudosimulation>force_field_stuff/"
    shadow: "minimal"
    log:
        log = f"<pseudosimulation>logging_gromacs.log"
    benchmark:
        repeat(f"<pseudosimulation>gromacs_benchmark.txt", 1)
    output:
        energy = f"<pseudosimulation>energy.xvg",
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f {input.runfile} -c {input.structure} -r {input.structure} -p {input.topology} -o result.tpr  -n {input.index}
        gmx22 mdrun -s result.tpr -rerun {input.trajectory} -g {log.log}
        gmx22 energy -f ener.edr -o {output.energy} < {input.select_energy}
        """

rule trajectory_slice_m1:
    input:
        structure = rules.trajectory_slice.output.frame_gro,
        index = "<pseudosimulation>index.ndx",
    output:
        trajectory = "{path}trajectory_slices/m1_frame_{frame_i}.gro",
    shadow: "minimal"
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "2\n" | gmx22 trjconv -f {input.structure} -n {input.index} -s {input.structure} -o {output.trajectory}
        """

rule trajectory_slice_com_m2:
    input:
        structure = rules.trajectory_slice.output.frame_gro,
        index = "<pseudosimulation>index.ndx",
    output:
        trajectory = "{path}trajectory_slices/COM_m2_frame_{frame_i}.gro",
    shadow: "minimal"
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "3\n" | gmx22 traj -f {input.structure} -s {input.structure} -n {input.index} -com -oxt {output.trajectory}
        """

rule full_trajectory_com_m2:
    input:
        structure = "{path}structure.gro",
        trajectory = "{path}trajectory.xtc",
        index = "{path}index.ndx",
    output:
        trajectory = "{path}COM_m2.xtc",
        positions = "{path}COM_m2.xvg",
    shadow: "minimal"
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "3\n" | gmx22 traj -f {input.trajectory} -s {input.structure} -n {input.index} -com -oxt {output.trajectory} -ox {output.positions}
        """

rule combine_m1_com_m2:
    input:
        structure_m1 = rules.trajectory_slice_m1.output.trajectory,
        structure_com_m2 = rules.trajectory_slice_com_m2.output.trajectory,
    output:
        structure = "{path}trajectory_slices/m1_COM_m2_frame_{frame_i}.gro",
    run:
        m1 = read_object(input.structure_m1)
        m2 = read_object(input.structure_com_m2)

        merged = Merge(m1.atoms,m2.atoms)
        merged.dimensions = m1.dimensions
        merged.atoms.write(output.structure)

rule structure_com_m2:
    input:
        structure = "{path}structure.gro",
        index = "<pseudosimulation>index.ndx",
    output:
        trajectory = "{path}COM_m2.gro",
    shadow: "minimal"
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "3\n" | gmx22 traj -f {input.structure} -s {input.structure} -n {input.index} -com -oxt {output.trajectory}
        """

rule structure_m1_com_m2:
    input:
        structure_m1 = "<pseudosimulation>molecule1.gro",
        structure_com_m2 = "{path}COM_m2.gro"
    output:
        structure = "{path}structure_COM.gro",
    run:
        m1 = read_object(input.structure_m1)
        m2 = read_object(input.structure_com_m2)

        merged = Merge(m1.atoms,m2.atoms)
        merged.dimensions = m1.dimensions
        merged.atoms.write(output.structure)

rule trajectory_centered_at_m2_COM:
    """
    Write the whole trajectory translated in such a way that the COM of molecule 2 is at (0,0,0) in each frame and
    molecule1 is not written. This is useful so we can later assign the best rotation.
    """
    input:
        trajectory = "{path}trajectory.xtc",
        structure="{path}structure.gro",
        index="{path}index.ndx",
    output:
        trajectory="{path}m2_trajectory_centered.xtc",
    shadow: "minimal"
    shell:
        """
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "3\n3\n" |  gmx22 trjconv  -s {input.structure}  -f {input.trajectory} -o {output.trajectory} -n {input.index} -center -boxcenter zero
        """
