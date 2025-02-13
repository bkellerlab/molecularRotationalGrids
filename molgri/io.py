import os
import re
import subprocess
from subprocess import PIPE, run
import MDAnalysis as mda
import pandas as pd
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import sparse
from scipy.constants import physical_constants
from MDAnalysis import Merge
from pathlib import Path

from molgri.molecules.pts import Pseudotrajectory
import MDAnalysis.transformations as trans

from molgri.space.translations import TranslationParser

HARTREE_TO_J = physical_constants["Hartree energy"][0]
AVOGADRO_CONSTANT = physical_constants["Avogadro constant"][0]


class GridWriter:
    """
    Takes in strings and writes out a grid.
    """

    def __init__(self, *args, **kwargs):
        from molgri.space.fullgrid import FullGrid
        self.fg = FullGrid(*args, **kwargs)

    def save_full_grid(self, path_grid_file: str):
        np.save(path_grid_file, self.fg.get_full_grid_as_array())

    def save_volumes(self, path_volumes: str):
        np.save(path_volumes, self.fg.get_total_volumes())

    def save_borders_array(self, path_borders_array: str):
        sparse.save_npz(path_borders_array, self.fg.get_full_borders())

    def save_distances_array(self, path_distances_array: str):
        sparse.save_npz(path_distances_array, self.fg.get_full_distances())

    def save_adjacency_array(self, path_adjacency_array: str):
        sparse.save_npz(path_adjacency_array, self.fg.get_full_adjacency())



class GridReader:
    """
    Loads files saved by GridWriter
    """

    def __init__(self):
        pass

    def load_full_grid(self, path_grid_file: str) -> NDArray:
        return np.load(path_grid_file)

    def load_volumes(self, path_volumes: str) -> NDArray:
        return np.load(path_volumes)

    def load_borders_array(self, path_borders_array: str) -> sparse.coo_array:
        return sparse.load_npz(path_borders_array)

    def load_distances_array(self, path_distances_array: str) -> sparse.coo_array:
        return sparse.load_npz(path_distances_array)

    def load_adjacency_array(self, path_adjacency_array: str) -> sparse.coo_array:
        return sparse.load_npz(path_adjacency_array)


class OneMoleculeReader:
    """
    Read a .gro or similar file that ony contains one molecule
    """

    def __init__(self, path_molecule: str, center_com=True):
        self.universe = mda.Universe(path_molecule)
        if center_com:
            self.universe.trajectory.add_transformations(trans.translate(-self.universe.atoms.center_of_mass()))

    def get_molecule(self) -> mda.Universe:
        return self.universe




class TwoMoleculeReader:
    """
    Read a .gro or similar file that contains exactly two molecules
    """

    def __init__(self, path_structure: str, path_trajectory: str):
        self.universe = mda.Universe(path_structure, path_trajectory)

    def get_full_pt(self) -> mda.Universe:
        return self.universe

    def get_only_second_molecule_pt(self, path_m2: str) -> mda.Universe:
        return self.universe.select_atoms(self._determine_second_molecule(path_m2))

    def _determine_second_molecule(self, path_m2: str) -> str:
        num_atoms_total = len(self.universe.atoms)
        m2 = OneMoleculeReader(path_m2).get_molecule()
        num_atoms_m2 = len(m2.atoms)
        # indexing in MDAnalysis is 1-based
        # we look for indices of the second molecule
        num_atoms_m1 = num_atoms_total - num_atoms_m2
        # indices are inclusive
        return f"bynum  {num_atoms_m1+1}:{num_atoms_total+1}"


class TwoMoleculeWriter:
    """
    Able to write structure of two molecules, but doesn't use grids
    """

    def __init__(self, path_molecule1: str, path_molecule2: str, cell_size_A: float):
        """
        This class writes a pseudotrajectory to a file. A PT consists of one molecule that is stationary at
        origin and one that moves with every time step. The fixed molecule is provided when the class is created
        and the mobile molecule as a generator when the method write_full_pt is called. Writing is done with
        MDAnalysis module, so all formats implemented there are supported.

        Args:
            name_to_save: base name of the PT file without paths or extensions
            parsed_central_molecule: a ParsedMolecule object describing the central molecule, will only be translated
                                     so that COM lies at (0, 0, 0) but not manipulated in any other way.
        """
        self.central_molecule = OneMoleculeReader(path_molecule1).get_molecule()
        self.moving_molecule = OneMoleculeReader(path_molecule2).get_molecule()
        self._center_both_molecules()
        self.dimensions = (cell_size_A, cell_size_A, cell_size_A, 90, 90, 90)

    def _center_both_molecules(self):
        com1 = self.central_molecule.atoms.center_of_mass()
        com2 = self.moving_molecule.atoms.center_of_mass()
        self.central_molecule.atoms.translate(-com1)
        self.moving_molecule.atoms.translate(-com2)

    def write_structure(self, start_distance_A: float, path_output_structure: str):
        """
        Write the one-frame topology file, eg in .gro format.
        """
        # translate the second one
        self.moving_molecule.atoms.translate([0, 0, float(start_distance_A)])

        # merge and write
        merged_u = Merge(self.central_molecule.atoms, self.moving_molecule.atoms)
        merged_u.dimensions = self.dimensions
        with mda.Writer(path_output_structure) as writer:
            writer.write(merged_u)


class PtWriter(TwoMoleculeWriter):

    def __init__(self, path_molecule1: str, path_molecule2: str, cell_size_A: float, path_grid: str):
        """
        This class writes a pseudotrajectory to a file. A PT consists of one molecule that is stationary at
        origin and one that moves with every time step. The fixed molecule is provided when the class is created
        and the mobile molecule as a generator when the method write_full_pt is called. Writing is done with
        MDAnalysis module, so all formats implemented there are supported.

        """
        super().__init__(path_molecule1, path_molecule2, cell_size_A)
        self.grid_array = GridReader().load_full_grid(path_grid)
        self.pt_universe = Pseudotrajectory(self.central_molecule, self.moving_molecule, self.grid_array).get_pt_as_universe()
        self.pt_universe.dimensions = self.dimensions
        self.n_atoms = len(self.central_molecule.atoms) + len(self.moving_molecule.atoms)

    def write_full_pt(self, path_output_pt: str, path_output_structure: str):
        """
        Write the trajectory file as well as the structure file (only at the first time-step).

        Args:
            path_output_pt: where trajectory should be saved
            path_output_structure: where topology should be saved
        """
        self.pt_universe.atoms.write(path_output_pt, frames='all')
        self.pt_universe.atoms.write(path_output_structure, frames=self.pt_universe.trajectory[[0]])

    def write_full_pt_in_directory(self, paths_trajectory: list, path_output_structure: str, extension_trajectory:
    str = "xyz"):
        """
        As an alternative to saving a full PT in a single trajectory file, you can create a directory with the same
        name and within it single-frame trajectories named with their frame index. Also save the structure file at
        first step.

            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            path_trajectory: where trajectory should be saved
            path_structure: where topology should be saved
        """
        #_create_dir_or_empty_it(directory_name)

        for i, ts in enumerate(self.pt_universe.trajectory):
            self.pt_universe.atoms.write(paths_trajectory[i])

        self.pt_universe.atoms.write(path_output_structure, frames=self.pt_universe.trajectory[[0]])


class EnergyReader:
    """
    Reads the .xvg file that gromacs outputs for energy.
    """

    def __init__(self, path_energy: str):
        self.path_energy = path_energy

    def load_energy(self) -> pd.DataFrame:
        if self.path_energy.endswith("xvg"):
            column_names = self._get_column_names()
            # skip 13 rows commented with # and then also a variable amount of rows commented with @
            table = pd.read_csv(self.path_energy, sep=r'\s+', comment='@', skiprows=13, header=None, names=column_names)
        elif self.path_energy.endswith("csv"):
            table = pd.read_csv(self.path_energy, index_col=0)
        else:
            raise ValueError(f"Don't know how to read the energy file type: {self.path_energy}")
        return table

    def _get_column_names(self) -> list:
        result = ["Time [ps]"]
        with open(self.path_energy, "r") as f:
            for line in f:
                # parse column number
                for i in range(0, 10):
                    if line.startswith(f"@ s{i} legend"):
                        split_line = line.split('"')
                        result.append(split_line[-2])
                if not line.startswith("@") and not line.startswith("#"):
                    break
        return result

    def load_single_energy_column(self, energy_type: str) -> NDArray:
        return self.load_energy()[energy_type].to_numpy()


def _create_dir_or_empty_it(directory_name):
    """
    Helper function that determines the name of the directory in which single frames of the trajectory are
    saved. If the directory already exists, its previous contents are deleted.

    Returns:
        path to the directory
    """
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        # delete contents if folder already exist
        filelist = [f for f in os.listdir(directory_name)]
        for f in filelist:
            os.remove(os.path.join(directory_name, f))


class BenchmarkReader:
    """
    Reads snakemake benchmark and gives the most importnat thing (time in s)
    """

    def __init__(self, path_benchmark_file: str):
        self.benchmark = pd.read_csv(path_benchmark_file, delimiter="\t")

    def get_time_in_s(self) -> pd.Series:
        return self.benchmark["s"]

    def get_mean_time_in_s(self) -> float:
        """
        Useful both to get the mean and also to get the only time as a float if there is just one line in the file.

        Returns: Time in the seconds
        """
        return self.get_time_in_s().mean()


class ParameterReader:
    """
    Reads the .yaml files we use to set-up calculations with snakemake.
    """

    def __init__(self, path_parameter_file: str):
        with open(path_parameter_file, 'r') as f:
            self.parameters = yaml.safe_load(f)

    def get_all_params_as_dict(self) -> dict:
        return self.parameters

    def get_grid_params_als_dict(self) -> dict:
        return self.parameters["params_grid"]

    def get_grid_size_params_als_dict(self) -> tuple:
        grid_params = self.get_grid_params_als_dict()
        # keep only the three sizes of grids, the rest we don#t care about

        num_directions = grid_params["num_directions"]
        num_orientations = grid_params["num_orientations"]
        num_radii = TranslationParser(grid_params["radial_distances_nm"]).get_N_trans()

        return (num_directions, num_orientations, num_radii)


class ItsReader:
    """
    Read the .csv files of implied timescales, be able to recognise some might be illogical etc.
    """

    def __init__(self, path_its_file: str):
        self.its = pd.read_csv(path_its_file)

    def get_first_N_its(self, N: int = 5) -> tuple:
        return tuple(self.its.iloc[0][:N])



class QuantumMolecule:

    """
    Just a simple class to collect variables that define a QM molecule.
    """

    def __init__(self, charge: int, multiplicity: int, xyz_file: str,
                 fragment_1_len: int = None, fragment_2_len: int = None):
        self.charge = charge
        self.multiplicity = multiplicity
        self.fragment_1_len = fragment_1_len
        self.fragment_2_len = fragment_2_len
        self.xyz_file = xyz_file


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


class OrcaWriter:

    """
    This class builds orca input files specifically for our typical set-up of two molecules.
    """

    def __init__(self, molecule:  QuantumMolecule, set_up: QuantumSetup):
        self.molecule = molecule
        self.setup = set_up
        with open(self.molecule.xyz_file, "r") as f:
            self.xyz_file_lines = f.readlines()
        self.total_text = ""

    def write_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(self.total_text)
        self.total_text = ""

    def _write_first_line(self, geo_optimization: bool = False):
        """
        First line looks something like this: ! PBE0 D4 def2-tzvp Opt <- depends on functional, SP/optimization and
        basis set.


        Args:
            geo_optimization (bool): if True option Opt will be selected, else SP
        """
        if geo_optimization:
            optimization_str = "Opt"
        else:
            optimization_str = ""

        self.total_text += f"! {self.setup.functional} {self.setup.dispersion_correction} {self.setup.basis_set} {optimization_str}\n"

    def _write_fragment_constraint(self):
        """

        Returns:

        """
        assert self.molecule.fragment_1_len is not None, "Need to know fragment lengths to constrain them!"
        assert self.molecule.fragment_2_len is not None, "Need to know fragment lengths to constrain them!"


        self.total_text += f"""
%geom


    ConnectFragments
     {{1 2 C}}      # constrain the internal coordinates
              #  connecting fragments 1 and 2
    end

    Fragments
      1 {{0:{self.molecule.fragment_1_len-1}}} end
      2 {{{self.molecule.fragment_1_len}:{self.molecule.fragment_1_len+self.molecule.fragment_2_len-1}}} end
    end

end\n"""

    def _write_solvent(self):
        if self.setup.solvent is not None:
            self.total_text += "%CPCM SMD TRUE\n"
            self.total_text += f'SMDSOLVENT "{self.setup.solvent}"\n'
            self.total_text += "END\n"

    def _write_resources(self):
        # limit the number of SCF cycles to make hopeless calculations fail quickly
        if self.setup.num_scf is not None:
            self.total_text += f"""
%scf
    MaxIter {self.setup.num_scf}
end\n"""
        if self.setup.num_cores is not None and self.setup.num_cores != "None":
            self.total_text += f"%PAL NPROCS {self.setup.num_cores} END\n"
        if self.setup.ram_per_core is not None and self.setup.ram_per_core != "None":
            self.total_text += f"%maxcore {self.setup.ram_per_core}\n"

    def make_entire_trajectory_inp(self, geo_optimization: bool, constrain_fragments: bool = False):
        num_atoms = self.molecule.fragment_1_len + self.molecule.fragment_2_len
        len_segment_pt = num_atoms + 2
        len_segment_pt = num_atoms+2
        len_pt_file = len(self.xyz_file_lines) - 1
        len_trajectory = len_pt_file // len_segment_pt
        for i in range(len_trajectory):
            # all that comes before molecule
            self._write_first_line(geo_optimization=geo_optimization)
            self._write_solvent()
            self._write_resources()
            # writing this *xyz frame, don't need the num of atoms
            start_line = i * len_segment_pt + 2
            end_line = i * len_segment_pt + len_segment_pt
            self._write_molecule_specification("".join(self.xyz_file_lines[start_line:end_line]))
            # all that comes after
            if constrain_fragments:
                self._write_fragment_constraint()


    def _write_molecule_specification(self, use_string):
        """
        Here we don't reference the .xyz file but write coordinates directly into .inp.
        """
        self.total_text += f"* xyz {self.molecule.charge} {self.molecule.multiplicity}\n"
        self.total_text += use_string
        self.total_text += "*\n"

    def make_input(self, geo_optimization: bool = False, constrain_fragments: bool = False):
        self._write_first_line(geo_optimization=geo_optimization)
        self._write_solvent()
        self._write_resources()
        self._write_molecule_specification()
        if constrain_fragments:
            self._write_fragment_constraint()



class OrcaReader:

    """
    Does not read .inp, but .out files
    """

    def __init__(self, out_file_path: str, is_multi_out: bool = False):
        self.out_file_path = out_file_path
        path = os.path.normpath(self.out_file_path)
        split_path = path.split(os.sep)
        self.frame_num = split_path[-2]
        self.calculation_directory = os.path.join(*split_path[:-1])
        self.is_multi_out = is_multi_out

    def assert_normal_finish(self, throw_error=True):
        """
        Make sure that the orca calculation finished normally.

        Args:
            throw_error (bool): if True, raise an error, if False, print a warning

        Either throws an error or prints a warning.
        """
        returncode = subprocess.run(f"""grep -q "****ORCA TERMINATED NORMALLY****" {self.out_file_path}""",
                       capture_output=True, shell=True).returncode

        if returncode != 0 and throw_error:
            raise ChildProcessError(f"Orca did not terminate normally; see {self.out_file_path }")
        elif returncode != 0 and not throw_error:
            return False
        return True

    def assert_optimization_complete(self, throw_error=True):
        """
        Make sure that the optimization is complete (not the same as normal finish!). Only relevant for optimizations.

        Args:
            throw_error (bool): if True, raise an error, if False, print a warning

        Either throws an error or prints a warning.
        """
        message_has_converged = """
***********************HURRAY********************
***        THE OPTIMIZATION HAS CONVERGED     ***
*************************************************
    """

        command = ['grep', f'"{message_has_converged}"', f"{self.out_file_path}"]
        returncode = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True).returncode
        if returncode != 0 and throw_error:
            raise ChildProcessError(f"Orca may have finished normally but did not converge; see {self.out_file_path}")
        elif returncode != 0 and not throw_error:
            print("Opt not complete")
            return False
        # also fail optimization if the calculation failed
        elif not self.assert_normal_finish(throw_error=False):
            print("No normal finish")
            return False
        return True

    def extract_time_orca_output(self) -> pd.Timedelta:
        """
        Take any orca output file and give me the time needed for the calculation.

        Args:
            output_file (str): path to the .out file

        Returns:
            Time as days hours:min:sec
        """
        line_time = subprocess.run(f"""grep "^TOTAL RUN TIME:" {self.out_file_path} | sed 's/^TOTAL RUN TIME: //'""",
                                   capture_output=True, text=True, shell=True)
        try:
            line_time = line_time.stdout.strip()
            line_time= line_time.replace("msec", "ms")
            time_h_m_s = line_time
        except AttributeError:
            time_h_m_s = 0

        return time_h_m_s

    def extract_energy_orca_output(self) -> list:
        """
        Take any orca output file and give me the total energy resulting from the calculation.

        Returns:
            Energy in the unit of Hartrees
        """
        # need the last one so use tail
        line_energy = subprocess.run(f'grep "^FINAL SINGLE POINT ENERGY" {self.out_file_path} | tail -n 1  | sed '
                                     f'"s/^FINAL SINGLE POINT '
                                     f'ENERGY //"', shell=True,
                                     capture_output=True, text=True)
        try:
            print(line_energy)
            line_energy = line_energy.stdout.strip()
            energy_hartree = float(line_energy)
        except ValueError:
            energy_hartree = np.NaN

        return energy_hartree

    def extract_num_atoms(self):
        line = subprocess.run(
            f'grep "^Number of atoms" {self.out_file_path}| head -n 1 ',
            shell=True, capture_output=True, text=True)
        number_atoms = line.stdout
        number_atoms = int(number_atoms.strip().split()[-1])
        return number_atoms

    def extract_optimized_xyz(self) -> str:
        if self.assert_optimization_complete(throw_error=False):
            # find the line number with the last occurence of CARTESIAN COORDINATES (ANGSTROEM)
            line = subprocess.run(
                f'grep -n "CARTESIAN COORDINATES (ANGSTROEM)" {self.out_file_path} | cut -d: -f1 | tail -n 1 ',
                shell=True, capture_output=True, text=True)
            line_number_last_coo = int(line.stdout)
            # start two lines after that, finish two lines + molecule length later
            start_point = 2 + line_number_last_coo
            end_point = 2 + line_number_last_coo + self.extract_num_atoms() -1
            command = ['head', '-n', f"{end_point}", f"{self.out_file_path}", "|", "tail", "-n", f"+{start_point}"]

            line = subprocess.run(
                f'head -n {end_point} {self.out_file_path} | tail -n +{start_point}',
                shell=True, capture_output=True, text=True)

            # starting with num of atoms and comment line
            result = f"{self.extract_num_atoms()}\n"
            result += "\n"
            result += line.stdout
            return result
        else:
            # to indicate an error while preserving pt length the initial structure is copied but all element names are
            # changed to X
            print(f"Not complete {self.out_file_path}")
            return ""

    def extract_last_coordinates_from_opt(self) -> str:
        """
        Extract the last structure of the optimization.
        """
        # try to find _trj.xyz in the directory
        for file in os.listdir(self.calculation_directory):
            if str(file).endswith("_trj.xyz"):
                orca_traj_xyz_file = os.path.join(self.calculation_directory, file)
                line_number_last_coo = subprocess.run(
                    f"""grep -n "Coordinates from" {orca_traj_xyz_file} | tail -n 1 | cut -d: -f1""",
                    shell=True, capture_output=True).stdout
                line_with_num_of_atoms = int(line_number_last_coo) - 1
                command = ['tail', '-n', f"+{line_with_num_of_atoms}", f"{orca_traj_xyz_file}"]
                result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
                return result.stdout
        else:
            raise FileNotFoundError(f"Cannot find any _trj.xyz file in {self.calculation_directory}")

    def extract_last_coordinates_to_file(self, file_path: str):
        """
        Same as extract_last_coordinates_from_opt, but immediately write to a file.

        Args:
            file_path (str): a path where the new file should be
        """
        file_contents = self.extract_last_coordinates_from_opt()
        with open(file_path, "w") as f:
            f.write(file_contents)

    def get_frame_num(self):
        return int(self.frame_num)


def read_multi_out_into_csv(multi_out: str, csv_file_to_write: str, setup: QuantumSetup, size_per_batch: int):
    """
    Read a list of orca .out files that were created with the same set-up (functional, basis set ...). Save the
    energies and generation times. Times can optionally be read from the benchmark files

    Args:
        out_files_to_read (list): a list of paths, usually to a number of .out files calculated along a molgri pt
        csv_file_to_write (str): a path to a csv file where the data will be recorded.
        setup ():

    Returns:

    """

    columns = ["File", "Frame", "Functional", "Basis set", "Dispersion correction", "Solvent",
               "Energy [hartree]"]

    all_df = []


    for out_file_to_read in multi_out:
        my_reader = OrcaReader(out_file_to_read, is_multi_out=True)
        energy_hartree = my_reader.extract_energy_orca_output()

        num_structures = subprocess.run(f"grep -oP 'There are \K\d+(?= structures to be calculated)' {out_file_to_read}",
                                        shell=True, capture_output=True)
        num_structures = int(num_structures.stdout)
        my_path = Path(out_file_to_read)
        batch_index = int(list(my_path.parts)[-2].split("_")[1])
        frames = list(range(batch_index*size_per_batch, batch_index*size_per_batch+np.min([num_structures, size_per_batch])))
        print(frames)


        all_data = []
        for energy, frame in zip(energy_hartree, frames):
            all_data.append([out_file_to_read, frame, setup.functional, setup.basis_set, setup.dispersion_correction,
                         setup.solvent, energy])

        df = pd.DataFrame(all_data, columns=columns)
        df["Energy [kJ/mol]"] = df["Energy [hartree]"] / 1000.0 * (HARTREE_TO_J * AVOGADRO_CONSTANT)

        df["Normal Finish"] = my_reader.assert_normal_finish(throw_error=False)
        df["Optimization Complete"] = my_reader.assert_optimization_complete(throw_error=False)
        all_df.append(df)

    combined_df = pd.concat(all_df)
    combined_df.to_csv(csv_file_to_write, index=False)


def read_important_stuff_into_csv(out_files_to_read: list, csv_file_to_write: str, setup: QuantumSetup,
                                  num_points: int, is_pt=True):
    """
    Read a list of orca .out files that were created with the same set-up (functional, basis set ...). Save the
    energies and generation times. Times can optionally be read from the benchmark files

    Args:
        out_files_to_read (list): a list of paths, usually to a number of .out files calculated along a molgri pt
        csv_file_to_write (str): a path to a csv file where the data will be recorded.
        setup ():

    Returns:

    """

    columns = ["File", "Frame", "Functional", "Basis set", "Dispersion correction", "Solvent",
               "Energy [hartree]", "Time [h:m:s]", "Normal Finish", "Optimization Complete"]

    all_df = []

    from pathlib import Path


    all_frame_indices = [int(Path(out_file).parts[-2]) for out_file in out_files_to_read]

    for frame_index in range(num_points):
        if frame_index in all_frame_indices:
            out_file_to_read = out_files_to_read[all_frame_indices.index(frame_index)]
            my_reader = OrcaReader(out_file_to_read)
            energy_hartree = my_reader.extract_energy_orca_output()
            time_h_m_s = my_reader.extract_time_orca_output()
            normal_finish = my_reader.assert_normal_finish(throw_error=False)
            optimization_complete = my_reader.assert_optimization_complete(throw_error=False)
        else:
            out_file_to_read = np.NaN
            energy_hartree = np.NaN
            time_h_m_s = np.NaN
            normal_finish = False
            optimization_complete = False

        all_data = [[out_file_to_read, frame_index, setup.functional, setup.basis_set, setup.dispersion_correction,
                     setup.solvent, energy_hartree, time_h_m_s, normal_finish, optimization_complete]]

        df = pd.DataFrame(all_data, columns=columns)
        df["Energy [kJ/mol]"] = df["Energy [hartree]"] / 1000.0 * (HARTREE_TO_J * AVOGADRO_CONSTANT)
        all_df.append(df)

    # for i, out_file_to_read in enumerate(out_files_to_read):
    #     my_reader = OrcaReader(out_file_to_read)
    #     energy_hartree = my_reader.extract_energy_orca_output()
    #
    #     if is_pt:
    #         frame = my_reader.get_frame_num()
    #     else:
    #         frame = None
    #
    #     time_h_m_s = my_reader.extract_time_orca_output()
    #
    #     all_data = [[out_file_to_read, frame, setup.functional, setup.basis_set, setup.dispersion_correction,
    #                  setup.solvent, energy_hartree, time_h_m_s]]
    #
    #     df = pd.DataFrame(all_data, columns=columns)
    #     df["Energy [kJ/mol]"] = df["Energy [hartree]"] / 1000.0 * (HARTREE_TO_J * AVOGADRO_CONSTANT)
    #
    #     df["Normal Finish"] = my_reader.assert_normal_finish(throw_error=False)
    #     df["Optimization Complete"] = my_reader.assert_optimization_complete(throw_error=False)
    #     all_df.append(df)

    combined_df = pd.concat(all_df)
    try:
        combined_df["Time [h:m:s]"] = pd.to_timedelta(combined_df["Time [h:m:s]"])
        combined_df["Time [s]"] = np.where(combined_df["Normal Finish"], combined_df["Time [h:m:s]"].dt.total_seconds(), np.NaN)
    except:
        # THIS DELTA TIME THING IS A HEADACHE!!!!
        pass
    combined_df.to_csv(csv_file_to_write, index=False)


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

