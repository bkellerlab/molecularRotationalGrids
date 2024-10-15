import os
import MDAnalysis as mda
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from MDAnalysis import Merge

from molgri.molecules.pts import Pseudotrajectory

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

    def load_molecule(self, path_molecule: str) -> mda.Universe:
        return mda.Universe(path_molecule)


class TwoMoleculeReader(OneMoleculeReader):
    """
    Read a .gro or similar file that ony contains one molecule
    """

    def load_full_pt(self, path_structure: str, path_trajectory: str):
        return mda.Universe(path_structure, path_trajectory)


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
        self.central_molecule = OneMoleculeReader().load_molecule(path_molecule1)
        self.moving_molecule = OneMoleculeReader().load_molecule(path_molecule2)
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

        Args:
            name_to_save: base name of the PT file without paths or extensions
            parsed_central_molecule: a ParsedMolecule object describing the central molecule, will only be translated
                                     so that COM lies at (0, 0, 0) but not manipulated in any other way.
        """
        super().__init__(path_molecule1, path_molecule2, cell_size_A)
        self.grid_array = GridReader().load_full_grid(path_grid)
        self.pt = Pseudotrajectory(self.moving_molecule, self.grid_array)
        self.n_atoms = len(self.central_molecule.atoms) + len(self.moving_molecule.atoms)

    def _merge_and_write(self, writer: mda.Writer):
        """
        Helper function to merge Atoms from central molecule with atoms of the moving molecule (at current positions)

        Args:
            writer: an already initiated object writing to file (eg a .gro or .xtc file)
            pt: a Pseudotrajectory object with method .get_molecule() that returns current ParsedMolecule
        """
        merged_universe = Merge(self.central_molecule.atoms, self.pt.get_molecule().atoms)
        merged_universe.dimensions = self.dimensions
        writer.write(merged_universe)

    def write_full_pt(self, path_output_pt: str, path_output_structure: str):
        """
        Write the trajectory file as well as the structure file (only at the first time-step).

        Args:
            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            path_trajectory: where trajectory should be saved
            path_structure: where topology should be saved
        """
        trajectory_writer = mda.Writer(path_output_pt, n_atoms=self.n_atoms, multiframe=True)
        last_i = 0
        for i, _ in self.pt.generate_pseudotrajectory():
            # for the first frame write out topology
            if i == 0:
                distance = np.linalg.norm(self.grid_array[i][:3])
                self.write_structure(start_distance_A=distance, path_output_structure=path_output_structure)
            self._merge_and_write(trajectory_writer)
            last_i = i
        product_of_grids = len(self.pt.get_full_grid())
        assert last_i + 1 == product_of_grids, f"Length of PT not correct, {last_i}=/={product_of_grids}"
        trajectory_writer.close()

    def write_full_pt_in_directory(self, path_output_pt: str, path_output_structure: str):
        """
        As an alternative to saving a full PT in a single trajectory file, you can create a directory with the same
        name and within it single-frame trajectories named with their frame index. Also save the structure file at
        first step.

            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            path_trajectory: where trajectory should be saved
            path_structure: where topology should be saved
        """
        directory_name, extension_trajectory = os.path.splitext(path_output_pt)
        _create_dir_or_empty_it(directory_name)
        for i, _ in self.pt.generate_pseudotrajectory():
            if i == 0:
                distance = np.linalg.norm(self.grid_array[i][:3])
                self.write_structure(start_distance_A=distance, path_output_structure=path_output_structure)
            f = f"{directory_name}/{i}{extension_trajectory}"
            with mda.Writer(f) as structure_writer:
                self._merge_and_write(structure_writer)


class EnergyReader:
    """
    Reads the .xvg file that gromacs outputs for energy.
    """

    def load_energy(self, path_energy: str) -> pd.DataFrame:
        column_names = self._get_column_names(path_energy)
        # skip 13 rows commented with # and then also a variable amount of rows commented with @
        table = pd.read_csv(path_energy, sep=r'\s+', comment='@', skiprows=13, header=None, names=column_names)
        return table

    def _get_column_names(self, path_energy: str) -> list:
        result = ["Time [ps]"]
        with open(path_energy, "r") as f:
            for line in f:
                # parse column number
                for i in range(0, 10):
                    if line.startswith(f"@ s{i} legend"):
                        split_line = line.split('"')
                        result.append(split_line[-2])
                if not line.startswith("@") and not line.startswith("#"):
                    break
        return result

    def load_single_energy_column(self, path_energy: str, energy_type: str) -> NDArray:
        return self.load_energy(path_energy)[energy_type].to_numpy()


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