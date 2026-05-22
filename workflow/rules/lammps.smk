
from MDAnalysis import Merge, Universe
import MDAnalysis as md
from MDAnalysis.coordinates.XYZ import XYZWriter
from MDAnalysis.topology.guessers import guess_masses

from workflow.helpers.io import get_num_atoms, read_object, write_object
#
# rule trajectory_slice:
#     """
#     We want to extract just the frame with index frame_i from a full trajectory.
#     """
#     input:
#         structure="<pseudosimulation>structure.<ext_str>",
#         trajectory="<pseudosimulation>trajectory.<ext_trj>",
#     output:
#         frame_gro="<pseudosimulation>trajectory_slices/frame_{frame_i}.<ext_str>",
#     run:
#         N_atoms = get_num_atoms(input.structure)
#         block_len = N_atoms + 2
#         frame_index = int(wildcards.frame_i)
#
#         # which lines in the file to read
#         start=frame_index*block_len +1
#         end=(frame_index+1)*block_len
#
#         shell(f"""sed -s -n {start},{end}p {input.trajectory} > {output.frame_gro}""")
#
# rule assignment_slice:
#     """
#     We want to extract just the frame with index frame_i from a full trajectory.
#     """
#     input:
#         structure="<pseudosimulation>structure.<ext_str>",
#         trajectory="<outputs_assignment>wrapped_trajectory.<ext_trj>",
#     output:
#         frame_gro="<outputs_assignment>trajectory_slices/frame_{frame_i}.<ext_str>",
#     run:
#         N_atoms = get_num_atoms(input.structure)
#         block_len = N_atoms + 2
#         frame_index = int(wildcards.frame_i)
#
#         # which lines in the file to read
#         start=frame_index*block_len +1
#         end=(frame_index+1)*block_len
#
#         shell(f"""sed -s -n {start},{end}p {input.trajectory} > {output.frame_gro}""")
#
# rule trajectory_slice_simulation:
#     """
#     We want to extract just the frame with index frame_i from a full trajectory.
#     """
#     input:
#         trajectory="<simulation>dump.lammpstrj",
#     output:
#         frame_gro="<simulation>trajectory_slices/frame_{frame_i}.<ext_str>",
#     run:
#         u = Universe(input.trajectory,format="LAMMPSDUMP")
#         u.trajectory[int(wildcards.frame_i)]
#         with XYZWriter(output.frame_gro,n_atoms=len(u.atoms)) as W:
#             W.write(u.atoms)


rule convert_to_lammps:
    input:
        trajectory = "<pseudosimulation>trajectory.<ext_trj>",
    output:
        lammps_trajectory = "<pseudosimulation>lammps_trajectory.dcd",
    run:
        u = Universe(input.trajectory)
        with md.coordinates.LAMMPS.DCDWriter(output.lammps_trajectory,n_atoms=u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                W.write(u.atoms)


rule complete_energy_to_energy:
    input:
        energy="{path}complete_energy.csv"
    output:
        energy="{path}energy.csv"
    params:
        every_ith = config["analysis"]["every_ith_energy"]
    run:
        df = read_object(input.energy)
        df_new = df.iloc[::int(params.every_ith)].reset_index(drop=True)
        write_object(df_new, output.energy)

rule copy_complete_energy:
    input:
        energy="<simulation>complete_energy.csv",
        index="<simulation>index.ndx",
        production="<simulation>production.mdp",
        structure="<simulation>structure.gro",
    output:
        energy="<outputs_assignment>complete_energy.csv",
        index="<pseudosimulation>index.ndx",
        production="<pseudosimulation>production.mdp",
        structure="<pseudosimulation>structure.gro",
    shell:
        """
        cp {input.energy} {output.energy}
        cp {input.index} {output.index}
        cp {input.production} {output.production}
        cp {input.structure} {output.structure}
        """

rule convert_to_xtc:
    input:
        structure="<pseudosimulation>structure.gro",
        trajectory="<simulation>dump.lammpstrj",
    output:
        structure="<simulation>structure.gro",
        trajectory="<simulation>dump.xtc"
    run:

        u = Universe(input.trajectory,format="LAMMPSDUMP")
        print(u.atoms[0].type)
        u.add_TopologyAttr('elements')
        u.add_TopologyAttr('names')
        dict_labels = {"1": "H", "2": "C", "3": "O", "4": "O", "5": "O", "6": "Zr", "OW": "O", "HW1": "H", "HW2": "H",
                       "7": "Ar", "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C", "CG": "C", "CZ": "C", "HD1": "H",
                       "HD2": "H", "HE1": "H", "HE2": "H", "HG": "H", "HZ": "H"}
        #initial_labels = u.atoms.names
        u.atoms.elements = [dict_labels[name] if name in dict_labels.keys() else name for name in u.atoms.types]
        u.atoms.names = [dict_labels[name] if name in dict_labels.keys() else name for name in u.atoms.types]
        u.atoms.types = u.atoms.names
        u.add_TopologyAttr('masses')
        u.atoms.masses = guess_masses(u.atoms.elements)
        u.atoms.write(output.structure)
        with md.coordinates.XTC.XTCWriter(output.trajectory,n_atoms=u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                W.write(u.atoms)

rule postprocess_dump:
    input:
        original_trajectory = f"<simulation>dump.xtc",
        #structure_tpr=f"<simulation>structure.tpr",
        structure_gro=f"<simulation>structure.gro",
        index=f"<simulation>index.ndx",
        runfile=f"<simulation>production.mdp",
    benchmark:
        repeat(f"<simulation>duration_gromacs_postprocessing.txt",1)
    shadow: "minimal"
    output:
        centered_trajectory = f"<simulation>centered_trajectory.xtc",
        trajectory=f"<simulation>trajectory.xtc",
    run:
        from workflow.helpers.io import read_from_mdrun

        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))
        timesteps = writeout * time_step_ps
        shell("""
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        # now fit to first frame
        echo "0\n" |  gmx22 trjconv -f {input.original_trajectory} -s {input.structure_gro} -pbc mol -boxcenter tric -o {output.centered_trajectory} -n {input.index}
        echo "4\n0\n" |  gmx22 trjconv -fit trans -f {output.centered_trajectory} -o {output.trajectory} -s  {input.structure_gro} -n {input.index}
        echo "0\n" |  gmx22 trjconv -f {output.trajectory} -s  {input.structure_gro} -o {output.trajectory} -n {input.index} -timestep {timesteps}
            """)

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
        structure_m1 = "<pseudosimulation>molecule1.xyz",
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
