"""
Everything here is saved to the pseudosimulation/gromacs folder: creating a structure, pseudotrajectory and calculating the energy along it.
"""
from MDAnalysis.topology.guessers import guess_masses

from workflow.helpers.io import read_object, write_object

rule copy_molecular_files_from_input:
    """
    Here the goal is just to start a new directory and copy molecular files there.
    """
    input:
        molecule_1 = f"<inputs_structures><molecule1>.<ext_inp>",
        molecule_2 = f"<inputs_structures><molecule2>.<ext_inp>",
    output:
        molecule_1 = f"<pseudosimulation>molecule1.<ext_inp>",
        molecule_2 = f"<pseudosimulation>molecule2.<ext_inp>",
    run:
        from molgri.molecules.bimolecular import move_to_center

        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)

        # center molecules
        m1 = move_to_center(m1)
        m2 = move_to_center(m2)

        write_object(m1, output.molecule_1)
        write_object(m2, output.molecule_2)



rule create_bulk_structure:
    input:
        molecule_1 = f"<pseudosimulation>molecule1.<ext_inp>",
        molecule_2 = f"<pseudosimulation>molecule2.<ext_inp>",
    output:
        structure = f"<pseudosimulation>bulk_structure.<ext_str>",
    run:
        from molgri.molecules.bimolecular import get_bimolecular_structure
        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)
        z_distance = 20
        structure = get_bimolecular_structure(m1, m2, z_distance=z_distance)
        write_object(structure, output.structure)



rule create_pseudotrajectory:
    """
    Here we are creating a pseudotrajectory from two molecules and a network.
    """
    input:
        structure = f"<pseudosimulation>structure.gro",
        molecule_1 = f"<pseudosimulation>molecule1.<ext_inp>",
        molecule_2 = f"<pseudosimulation>molecule2.<ext_inp>",
        network = f"<outputs_network>network.pkl"
    output:
        trajectory = f"<pseudosimulation>trajectory.<ext_trj>"
    run:
        from molgri.molecules.bimolecular import get_bimolecular_pseudotrajectory, move_to_center

        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)

        us = read_object(input.structure)
        m1.dimensions = us.dimensions
        m2.dimensions = us.dimensions
        network = read_object(input.network)
        weights = m2.atoms.masses
        coordinates = network.create_pseudotrajectory_coordinates_from(m2.atoms.positions, weights)
        pt = get_bimolecular_pseudotrajectory(m1, m2, coordinates)
        u=pt
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
        write_object(pt, output.trajectory)


rule read_in_energies:
    """
    Here we can assign the energy to the nodes of the network.
    """
    input:
        network = f"<outputs_network>network.pkl",
        energy = f"<pseudosimulation>energy.csv",
    output:
        network_energy = f"<pseudosimulation>network_energy.pkl"
    run:
        my_network = read_object(input.network)
        my_energy = read_object(input.energy)
        my_energy_array = my_energy["Energy [kJ/mol]"].to_numpy()
        my_network.add_node_properties(my_energy_array,"binding_energy")
        write_object(my_network, output.network_energy)
