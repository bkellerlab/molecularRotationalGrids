"""
Loading currently important examples locally.
"""
import pandas as pd
import numpy as np

from molgri.molecules.parsers import ParsedEnergy, FileParser, ParsedTrajectory


def load_simulation_data() -> ParsedTrajectory:
    path_energy = "/home/mdglasius/Modelling/trypsin_normal/nobackup/outputs/data.txt"
    df = pd.read_csv(path_energy)
    energies = df['Potential Energy (kJ/mole)'].to_numpy()[:, np.newaxis]
    pe = ParsedEnergy(energies=energies, labels=["Potential Energy"], unit="(kJ/mole)")
    pt_parser = FileParser(
             path_topology="/home/mdglasius/Modelling/trypsin_normal/inputs/trypsin_probe.pdb",
             path_trajectory="/home/mdglasius/Modelling/trypsin_normal/nobackup/outputs/aligned_traj.dcd")
    parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="segid B")
    parsed_trajectory.energies = pe
    return parsed_trajectory


def load_molgri_data() -> ParsedTrajectory:
    path_energy = "/home/mdglasius/Modelling/trypsin_test/output/measurements/temp_minimized.csv"
    df = pd.read_csv(path_energy)
    energies = df['potential'].to_numpy()[:, np.newaxis]
    pe = ParsedEnergy(energies=energies, labels=["Potential Energy"], unit="(kJ/mole)")
    pt_parser = FileParser(
             path_topology="/home/mdglasius/Modelling/trypsin_test/output/pt_files/final_trypsin_NH4_o_ico_512_b_zero_1_t_2153868773.gro",
             path_trajectory="/home/mdglasius/Modelling/trypsin_test/output/pt_files/minimizedPT.pdb")
    parsed_trajectory = pt_parser.get_parsed_trajectory(default_atom_selection="not protein")
    parsed_trajectory.energies = pe
    return parsed_trajectory