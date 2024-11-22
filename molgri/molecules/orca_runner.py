import subprocess
import os
from scipy.constants import physical_constants
import numpy as np
import re

HARTREE_TO_J = physical_constants["Hartree energy"][0]
AVOGADRO_CONSTANT = physical_constants["Avogadro constant"][0]

import pandas as pd


def nice_str_of(string: str):
    if not string:
        return "no"
    return re.sub(r'[^a-zA-Z0-9]', '', string)


class QuantumMolecule:

    """
    Just a simple class to collect variables that define a QM molecule.
    """

    def __init__(self, charge: int, multiplicity: int, path_xyz: str):
        self.charge = charge
        self.multiplicity = multiplicity
        self.path_xyz = path_xyz


class QuantumSetup:

    """
    Just a simple class to collect variables connected to QM calculation set-up (that are not molecule-specific).
    """

    def __init__(self, functional: str, basis_set: str, solvent: str = None, dispersion_correction: str = ""):
        self.functional = functional
        self.basis_set = basis_set
        self.solvent = solvent
        self.dispersion_correction = dispersion_correction


    def get_dir_name(self):
        return f"{nice_str_of(self.functional)}_{nice_str_of(self.basis_set)}_{nice_str_of(self.solvent)}_{nice_str_of(self.dispersion_correction)}/"


def make_inp_file(molecule: QuantumMolecule, setup: QuantumSetup, geo_optimization: str = "Opt") -> str:

    orca_input = f"! {setup.functional} {setup.dispersion_correction} {setup.basis_set} {geo_optimization}\n"

    if setup.solvent is not None:
        orca_input += "%CPCM SMD TRUE\n"
        orca_input += f'SMDSOLVENT "{setup.solvent}"\n'
        orca_input += "END\n"


    orca_input += f"*xyzfile {molecule.charge} {molecule.multiplicity} {molecule.path_xyz}\n"
    return orca_input


def read_important_stuff_into_csv(out_files_to_read: list, csv_file_to_write: str, setup: QuantumSetup):
    """
    Read different orca .out files that were created with the same set-up (functional, basis set ...). Save the
    energies and generation times

    Args:
        out_files_to_read ():
        csv_file_to_write ():
        setup ():

    Returns:

    """

    columns = ["File", "Functional", "Basis set", "Dispersion correction", "Solvent",
               "Time [h:m:s]", "Energy [hartree]"]

    all_df = []
    for out_file_to_read in out_files_to_read:
        energy_hartree, time_s = extract_energy_time_orca_output(out_file_to_read)
        all_data = np.array([[out_file_to_read, setup.functional, setup.basis_set, setup.dispersion_correction,
                              setup.solvent, time_s, energy_hartree]])
        df = pd.DataFrame(all_data, columns=columns)
        all_df.append(df)

    combined_df = pd.concat(all_df) #, ignore_index=True

    combined_df["Energy [kJ/mol]"] = HARTREE_TO_J * AVOGADRO_CONSTANT * combined_df["Energy [hartree]"] / 1000  # 1000 because kJ
    combined_df["Time [s]"] = combined_df["Time [h:m:s]"].dt.total_seconds()

    combined_df.to_csv(csv_file_to_write, index=False)

def assert_normal_finish(orca_output_file:str):
    returncode = subprocess.run(f'grep "****ORCA TERMINATED NORMALLY****" {orca_output_file}', shell=True,
                                text=False).returncode
    if returncode != 0:
        raise ChildProcessError(f"Orca did not terminate normally; see {orca_output_file}")


def extract_last_coordinates_from_opt(orca_traj_xyz_file: str, new_file: str):
    line_number_last_coo =subprocess.run(f"""grep -n "Coordinates from" {orca_traj_xyz_file} | tail -n 1 | cut -d: -f1""",
                                         shell=True, capture_output=True).stdout
    line_with_num_of_atoms = int(line_number_last_coo)-1
    subprocess.run(f"""tail -n +"{line_with_num_of_atoms}" {orca_traj_xyz_file} > {new_file}""",
                               shell=True)


class OrcaRun:

    def __init__(self, functional: str, basis_set: str, dimer_molecule: QuantumMolecule,
                 m1_molecule: QuantumMolecule = None, m2_molecule: QuantumMolecule = None, m1_equals_m2: bool = False,
                 solvent: str = None, dispersion_correction: str = ""):
        self.functional = functional
        self.basis_set = basis_set
        self.dimer_molecule = dimer_molecule
        self.m1_molecule = m1_molecule
        self.m2_molecule = m2_molecule
        self.m1_equals_m2 = m1_equals_m2
        self.solvent = solvent
        self.dispersion_correction = dispersion_correction

        self.start_dir = os.getcwd()
        # make a directory if not existing yet
        self.calc_dir = f"{functional}_{basis_set}_{solvent}_{dispersion_correction}/"
        if not os.path.isdir(self.calc_dir):
            os.mkdir(self.calc_dir)

    def make_input_file(self, which: str, geo_optimization: str = "Opt"):

        orca_input = f"! {self.functional} {self.dispersion_correction} {self.basis_set} {geo_optimization}\n"

        if self.solvent is not None:
            orca_input += "%CPCM SMD TRUE\n"
            orca_input += f'SMDSOLVENT "{self.solvent}"\n'
            orca_input += "END\n"

        match which:
            case "dimer":
                chosen_molecule = self.dimer_molecule
            case "m1":
                chosen_molecule = self.m1_molecule
            case "m2":
                chosen_molecule = self.m2_molecule
            case _:  # anything else
                raise ValueError(f"Unexpected value: which cannot be {which}")

        orca_input += f"*xyzfile {chosen_molecule.charge} {chosen_molecule.multiplicity} {chosen_molecule.path_xyz}\n"
        return orca_input




def extract_energy_time_orca_output(output_file: str) -> tuple:
    """
    Take any orca output file and give me the total energy and time needed for the calculation.

    Args:
        output_file (str): path to the .out file

    Returns:
        DataFrame with desired properties read from the file
    """
    # need the last one so use tail
    line_energy = subprocess.run(f'grep "^FINAL SINGLE POINT ENERGY" {output_file} | tail -n 1 | sed "s/^FINAL SINGLE POINT '
                                 f'ENERGY //"', shell=True,
                                 capture_output=True, text=True)
    line_time = subprocess.run(f"""grep "^TOTAL RUN TIME:" {output_file} | sed 's/^TOTAL RUN TIME: //'""", shell=True,
                               capture_output=True,
                               text=True)
    try:
        energy_hartree = float(line_energy.stdout.strip())
    except ValueError:
        energy_hartree = np.NaN
    line_time = line_time.stdout.strip()
    line_time= line_time.replace("msec", "ms")
    time_s = pd.to_timedelta(line_time)

    return energy_hartree, time_s

