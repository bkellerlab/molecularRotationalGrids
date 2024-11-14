import os
import MDAnalysis as mda
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from MDAnalysis import Merge

from molgri.molecules.pts import Pseudotrajectory
import MDAnalysis.transformations as trans


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

        for i, ts in enumerate(self.pt_universe.trajectory):
            self.pt_universe.atoms.write(f"{directory_name}/{str(i).zfill(10)}{extension_trajectory}")

        self.pt_universe.atoms.write(path_output_structure, frames=self.pt_universe.trajectory[[0]])


class EnergyReader:
    """
    Reads the .xvg file that gromacs outputs for energy.
    """

    def __init__(self, path_energy: str):
        self.path_energy = path_energy

    def load_energy(self) -> pd.DataFrame:
        column_names = self._get_column_names()
        # skip 13 rows commented with # and then also a variable amount of rows commented with @
        table = pd.read_csv(self.path_energy, sep=r'\s+', comment='@', skiprows=13, header=None, names=column_names)
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