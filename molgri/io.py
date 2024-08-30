"""
Writers and readers for different objects (grids, pseudotrajectories ...) Inputs to methods should be PATHS and
PARAMETERS.
"""
import hashlib
import numbers
from abc import ABC
import os
from ast import literal_eval
from functools import wraps

import MDAnalysis as mda
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from MDAnalysis import Merge
from scipy import sparse

# lots of imports just for typing
from molgri.constants import ALL_GRID_ALGORITHMS, DEFAULT_ALGORITHM_B, DEFAULT_ALGORITHM_O, NM2ANGSTROM, \
    ZERO_ALGORITHM_3D, \
    ZERO_ALGORITHM_4D
from molgri.molecules.pts import Pseudotrajectory
from molgri.space.fullgrid import FullGrid
from molgri.space.rotobj import SphereGrid3DFactory, SphereGrid4DFactory


class AbstractWriter(ABC):

    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        _create_dir_or_empty_it(self.output_folder)

    def write_all(self, **kwargs):
        """
        Save everything that should be saved.

        This method assumes that every method beginning with save_ is a saving function.
        """
        object_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name)) and method_name.startswith("save_")]
        for method in object_methods:
            saving_method = getattr(self, method)
            saving_method(**kwargs)


class AbstractReader(ABC):

    def __init__(self, output_folder: str):
        self.output_folder = output_folder

    def read_all(self, **kwargs):
        """
        Read everything that has been saved.

        This method assumes that every method beginning with save_ is a saving function.
        """
        object_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name)) and method_name.startswith("load_")]
        for method in object_methods:
            saving_method = getattr(self, method)
            saving_method(**kwargs)


class RotObjWriter(AbstractWriter):

    def __init__(self, rot_obj_name: str, is_3d: bool, output_folder: str):
        super().__init__(output_folder)
        self.is_3d = is_3d
        rotobj_name = RotObjParser(rot_obj_name, is_3d=is_3d)
        if is_3d:
            self.rotobj = SphereGrid3DFactory.create(alg_name=rotobj_name.get_alg(), N=rotobj_name.get_N())
        else:
            self.rotobj = SphereGrid4DFactory.create(alg_name=rotobj_name.get_alg(), N=rotobj_name.get_N())
        self.rotobj_voronoi = self.rotobj.get_spherical_voronoi()

    def save_rotobj_array(self):
        np.save(f"{self.output_folder}/array.npy", self.rotobj.get_grid_as_array())

    def save_borders_array(self):
        sparse.save_npz(f"{self.output_folder}/borders.npz", self.rotobj_voronoi.get_cell_borders())

    def save_distances_array(self):
        sparse.save_npz(f"{self.output_folder}/distances.npz", self.rotobj_voronoi.get_center_distances())

    def save_adjacency_array(self):
        sparse.save_npz(f"{self.output_folder}/adjancency.npz", self.rotobj_voronoi.get_voronoi_adjacency())

    def save_volumes(self):
        np.save(f"{self.output_folder}/volumes.npy", self.rotobj_voronoi.get_voronoi_volumes())


class RotObjReader(AbstractReader):

    def __init__(self, output_folder: str):
        super().__init__(output_folder)
        self.array = self._load_rotobj_array()
        self.adjacency = self._load_adjacency_array()
        self.borders = self._load_borders_array()
        self.distances = self._load_distances_array()
        self.volumes = self._load_volumes()

    def _load_rotobj_array(self):
        return np.load(f"{self.output_folder}/array.npy")

    def _load_borders_array(self):
        return sparse.load_npz(f"{self.output_folder}/borders.npz")

    def _load_distances_array(self):
        return sparse.load_npz(f"{self.output_folder}/distances.npz")

    def _load_adjacency_array(self):
        return sparse.load_npz(f"{self.output_folder}/adjancency.npz")

    def _load_volumes(self):
        return np.load(f"{self.output_folder}/volumes.npy")



class RotObjParser:

    def __init__(self, name_string: str, is_3d: bool):
        self.name_string = name_string
        self.is_3d = is_3d

    def get_N(self) -> int or None:
        """
        Try to find an integer representing number of grid points anywhere in the name.

        Returns:
            the number of points as an integer, if it can be found, else None

        Raises:
            ValueError if more than one integer present in the string (e.g. 'ico_12_17')
        """
        split_string = self.name_string.split("_")
        candidates = []
        for fragment in split_string:
            if fragment.isnumeric():
                candidates.append(int(fragment))
        # >= 2 numbers found in the string
        if len(candidates) > 1:
            raise ValueError(f"Found two or more numbers in grid name {self.name_string},"
                             f" can't determine num of points.")
        # exactly one number in the string -> return it
        elif len(candidates) == 1:
            return candidates[0]
        # no number in the string -> return None
        else:
            raise ValueError(f"No number in the provided string: {self.name_string}")

    def get_alg(self) -> str:
        # if number is 0 or 1, immediately return zero-alg
        if self.get_N() == 0 or self.get_N() == 1:
            if self.is_3d:
                return ZERO_ALGORITHM_3D
            else:
                return ZERO_ALGORITHM_4D
        split_string = self.name_string.split("_")
        candidates = []
        for fragment in split_string:
            if fragment in ALL_GRID_ALGORITHMS:
                candidates.append(fragment)
        # >= 2 algorithms found in the string
        if len(candidates) > 1:
            raise ValueError(f"Found two or more algorithm names in grid name {self.name_string}, can't decide.")
        # exactly one algorithm in the string -> return it
        elif len(candidates) == 1:
            return candidates[0]
        # no algorithm given -> select default
        else:
            # default for 3D
            if self.is_3d:
                return DEFAULT_ALGORITHM_O
            # default for 4D
            else:
                return DEFAULT_ALGORITHM_B


class TranslationWriter(AbstractWriter):

    """
    User input is expected in nanometers (nm)!

        Parse all ways in which the user may provide a linear translation grid. Currently supported formats:
            - a list of numbers, eg '[1, 2, 3]'
            - a linearly spaced list with optionally provided number of elements eg. 'linspace(1, 5, 50)'
            - a range with optionally provided step, eg 'range(0.5, 3, 0.4)'
    """

    def __init__(self, user_input: str, output_folder: str):
        """
        Args:
            user_input: a string in one of allowed formats
        """
        super().__init__(output_folder)
        self.user_input = user_input
        if "linspace" in self.user_input:
            bracket_input = self._read_within_brackets()
            self.trans_grid = np.linspace(*bracket_input, dtype=float)
        elif "range" in self.user_input:
            bracket_input = self._read_within_brackets()
            self.trans_grid = np.arange(*bracket_input, dtype=float)
        else:
            self.trans_grid = literal_eval(self.user_input)
            self.trans_grid = np.array(self.trans_grid, dtype=float)
            self.trans_grid = np.sort(self.trans_grid, axis=None)
        # all values must be non-negative
        assert np.all(self.trans_grid >= 0), "Distance from origin cannot be negative."
        # convert to angstrom
        self.trans_grid = self.trans_grid * NM2ANGSTROM

    def save_trans_grid(self):
        """Getter to access all distances from origin in angstorms."""
        np.save(f"{self.output_folder}/array.npy", self.trans_grid)

    def get_N_trans(self) -> int:
        """Get the number of translations in this grid."""
        return len(self.trans_grid)

    def _read_within_brackets(self) -> tuple:
        """
        Helper function to aid reading linspace(start, stop, num) and arange(start, stop, step) formats.
        """
        str_in_brackets = self.user_input.split('(', 1)[1].split(')')[0]
        str_in_brackets = literal_eval(str_in_brackets)
        if isinstance(str_in_brackets, numbers.Number):
            str_in_brackets = tuple((str_in_brackets,))
        return str_in_brackets


class TranslationReader(AbstractReader):

    def __init__(self, output_folder: str):
        super().__init__(output_folder)
        self.array = self.load_trans_grid()

    def load_trans_grid(self) -> NDArray:
        """Getter to access all distances from origin in angstorms."""
        return np.load(f"{self.output_folder}/array.npy")


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


class GridWriter(AbstractWriter):
    """
    Interprets the strings of user input, constructs the grid. Saves important information of a grid object.
    """

    def __init__(self, orientation_reader: RotObjReader, direction_reader: RotObjReader,
                 distance_reader: TranslationReader, output_folder: str):
        super().__init__(output_folder)
        self.full_grid = FullGrid(orientation_grid=orientation_reader, direction_grid=direction_reader,
                                  distance_grid=distance_reader)

    def save_full_grid(self):
        np.save(f"{self.output_folder}/array.npy", self.full_grid.get_full_grid_as_array())

    def save_volumes(self):
        np.save(f"{self.output_folder}/volumes.npy", self.full_grid.get_total_volumes())

    def save_borders_array(self):
        sparse.save_npz(f"{self.output_folder}/borders.npz", self.full_grid.get_full_borders())

    def save_distances_array(self):
        sparse.save_npz(f"{self.output_folder}/distances.npz", self.full_grid.get_full_distances())

    def save_adjacency_array(self):
        sparse.save_npz(f"{self.output_folder}/adjacency.npz", self.full_grid.get_full_adjacency())


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
