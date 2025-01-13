"""
An interface to Orca: writing input files, extracting data from output files, running calculations.
"""

import subprocess
import re


from scipy.constants import physical_constants
import numpy as np
import pandas as pd


HARTREE_TO_J = physical_constants["Hartree energy"][0]
AVOGADRO_CONSTANT = physical_constants["Avogadro constant"][0]


def nice_str_of(string: str) -> str:
    """
    Make a string "nice" (ready to use in file names etc) by removing all charcaters that are not alphanumeric.
    Special case: if input is an empty string, the output is "no" because that is easier to include in names.

    Args:
        string (str): input string to be cleaned up

    Returns:
        output string, same as input but without special characters
    """
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

    def __init__(self, functional: str, basis_set: str, solvent: str = None, dispersion_correction: str = "",
                 num_scf: int = 15, num_cores: int = None, ram_per_core: int = None):
        self.functional = functional
        self.basis_set = basis_set
        self.solvent = solvent
        self.dispersion_correction = dispersion_correction
        self.num_scf = num_scf
        self.num_cores = num_cores
        self.ram_per_core = ram_per_core

    def get_dir_name(self):
        """
        Calculations of this quantum set-up can be done in a folder that specifies the major settings.
        """
        return f"{nice_str_of(self.functional)}_{nice_str_of(self.basis_set)}_{nice_str_of(self.solvent)}_{nice_str_of(self.dispersion_correction)}/"


def make_inp_file(molecule: QuantumMolecule, setup: QuantumSetup, geo_optimization: bool = False) -> str:

    """
    We want to set up a .inp file for orca for any calculation we might wanna perform.

    Args:
        molecule (QuantumMolecule): object containing molecule-dependent parameters like charge, multiplicity
        setup (QuantumSetup): object containing molecule-independent parameters like basis set, functional
        geo_optimization (bool): True for optimization, False for single-point calculation

    Returns:
        The text of a full .inp file that can be immediately saved to a file.
    """

    if geo_optimization:
        orca_input = f"! {setup.functional} {setup.dispersion_correction} {setup.basis_set} Opt\n"
    else:
        orca_input = f"! {setup.functional} {setup.dispersion_correction} {setup.basis_set}\n"


    # limit the number of SCF cycles to make hopeless calculations fail quickly
    if setup.num_scf is not None:
        orca_input += f"""
%scf
    MaxIter {setup.num_scf}
end
"""
    if setup.num_cores is not None and setup.num_cores != "None":
        orca_input += f"%PAL NPROCS {setup.num_cores} END\n"
    if setup.ram_per_core is not None and setup.ram_per_core != "None":
        orca_input += f"%maxcore {setup.ram_per_core}\n"

    # put this last
    if setup.solvent is not None:
        orca_input += "%CPCM SMD TRUE\n"
        orca_input += f'SMDSOLVENT "{setup.solvent}"\n'
        orca_input += "END\n"

    orca_input += f"*xyzfile {molecule.charge} {molecule.multiplicity} {molecule.path_xyz}\n"
    return orca_input


def read_important_stuff_into_csv(out_files_to_read: list, csv_file_to_write: str, setup: QuantumSetup,
                                  benchmark_files_to_read: list = None):
    """
    Read a list of orca .out files that were created with the same set-up (functional, basis set ...). Save the
    energies and generation times. Times can optionally be read from the benchmark files

    Args:
        out_files_to_read (list): a list of paths, usually to a number of .out files calculated along a molgri pt
        csv_file_to_write (str): a path to a csv file where the data will be recorded.
        setup ():

    Returns:

    """

    columns = ["File", "Functional", "Basis set", "Dispersion correction", "Solvent",
               "Energy [hartree]", "Time [h:m:s]"]

    all_df = []

    for i, out_file_to_read in enumerate(out_files_to_read):
        energy_hartree = float(extract_energy_orca_output(out_file_to_read))

        # read time either from benchmark or from orca .out
        if benchmark_files_to_read is not None:
            benchmark_file_to_read = benchmark_files_to_read[i]
            time_s = float(pd.read_csv(benchmark_file_to_read, delimiter="\t")["s"][0])
            time_h_m_s = pd.to_timedelta(time_s, unit='s')
        else:
             time_h_m_s = extract_time_orca_output(out_file_to_read)

        all_data = [[out_file_to_read, setup.functional, setup.basis_set, setup.dispersion_correction,
                     setup.solvent, energy_hartree, time_h_m_s]]

        df = pd.DataFrame(all_data, columns=columns)
        df["Energy [kJ/mol]"] = df["Energy [hartree]"] / 1000.0 * (HARTREE_TO_J * AVOGADRO_CONSTANT)
        df["Time [s]"] = np.where(~df["Time [h:m:s]"].isna(), df["Time [h:m:s]"].dt.total_seconds(), np.NaN)
        all_df.append(df)

    combined_df = pd.concat(all_df)
    combined_df.to_csv(csv_file_to_write, index=False)


def assert_normal_finish(orca_output_file: str, throw_error=True):
    """
    Make sure that the orca calculation finished normally.

    Args:
        orca_output_file (str): path to a file where the orca output can be found
        throw_error (bool): if True, raise an error, if False, print a warning

    Either throws an error or prints a warning.

    """
    returncode = subprocess.run(f'grep "****ORCA TERMINATED NORMALLY****" {orca_output_file}', shell=True,
                                text=False).returncode
    if returncode != 0 and throw_error:
        raise ChildProcessError(f"Orca did not terminate normally; see {orca_output_file}")
    elif returncode != 0 and not throw_error:
        print(f"Orca did not terminate normally; see {orca_output_file}")


def extract_last_coordinates_from_opt(orca_traj_xyz_file: str, new_file: str):
    """


    Args:
        orca_traj_xyz_file ():
        new_file ():

    Returns:

    """
    line_number_last_coo =subprocess.run(f"""grep -n "Coordinates from" {orca_traj_xyz_file} | tail -n 1 | cut -d: -f1""",
                                         shell=True, capture_output=True).stdout
    line_with_num_of_atoms = int(line_number_last_coo)-1
    subprocess.run(f"""tail -n +"{line_with_num_of_atoms}" {orca_traj_xyz_file} > {new_file}""",
                               shell=True)


def extract_time_orca_output(output_file: str) -> pd.Timedelta:
    """
    Take any orca output file and give me the time needed for the calculation.

    Args:
        output_file (str): path to the .out file

    Returns:
        Time as days hours:min:sec
    """
    line_time = subprocess.run(f"""grep "^TOTAL RUN TIME:" {output_file} | sed 's/^TOTAL RUN TIME: //'""", shell=True,
                               capture_output=True, text=True)
    try:
        line_time = line_time.stdout.strip()
        line_time= line_time.replace("msec", "ms")
        time_h_m_s = pd.to_timedelta(line_time)
    except AttributeError:
        time_h_m_s = np.NaN

    return time_h_m_s


def extract_energy_orca_output(output_file: str) -> float:
    """
    Take any orca output file and give me the total energy resulting from the calculation.

    Args:
        output_file (str): path to the .out file

    Returns:
        Energy in the unit of Hartrees
    """
    # need the last one so use tail
    line_energy = subprocess.run(f'grep "^FINAL SINGLE POINT ENERGY" {output_file} | tail -n 1 | sed "s/^FINAL SINGLE POINT '
                                 f'ENERGY //"', shell=True,
                                 capture_output=True, text=True)
    try:
        energy_hartree = float(line_energy.stdout.strip())
    except ValueError:
        energy_hartree = np.NaN

    return energy_hartree

