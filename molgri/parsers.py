import hashlib
import numbers
import os
from pathlib import Path
from typing import Generator, Tuple, List

import numpy as np
from mendeleev.fetch import fetch_table
from ast import literal_eval

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import MDAnalysis as mda
from MDAnalysis.core import AtomGroup

from .constants import MOLECULE_NAMES, SIX_METHOD_NAMES, FULL_RUN_NAME
from .paths import PATH_OUTPUT_TRANSGRIDS


class NameParser:

    def __init__(self, name: str or dict):
        """
        Correct ordering: '[H2O_HF_]ico[_NO]_500_full_openMM[_extra]

        Args:
            string with a label used eg. for naming files, usually stating two molecules, grid type and size
        """
        # define all properties
        self.central_molecule = None
        self.rotating_molecule = None
        self.grid_type = None
        self.o_grid = None
        self.b_grid = None
        self.t_grid = None
        self.ordering = True
        self.num_grid_points = None
        self.traj_type = None
        self.open_MM = False
        self.is_real_run = False
        self.additional_data = None
        self.ending = None
        # parse name
        if isinstance(name, str):
            self._read_str(name)
        elif isinstance(name, dict):
            self._read_dict(name)

    def _read_str(self, name):
        try:
            if "." in name:
                name, self.ending = name.split(".")
        except ValueError:
            pass
        split_str = name.split("_")
        for split_item in split_str:
            if split_item in MOLECULE_NAMES:
                if self.central_molecule is None:
                    self.central_molecule = split_item
                else:
                    self.rotating_molecule = split_item
        for method_name in SIX_METHOD_NAMES:
            if method_name in split_str:
                self.grid_type = method_name
        # _o_, _b_ and _t_
        for i, sub_str in enumerate(split_str):
            if "o" == sub_str:
                self.o_grid = split_str[i+1]
            if "b" == sub_str:
                self.b_grid = split_str[i+1]
            if "t" == sub_str:
                self.t_grid = int(split_str[i+1])
        if "zero" in split_str:
            self.grid_type = "zero"
            self.num_grid_points = 1
        if "_NO" in name:
            self.ordering = False
        for split_item in split_str:
            if split_item.isnumeric():
                self.num_grid_points = int(split_item)
                break
        for traj_type in ["circular", "full"]:
            if traj_type in split_str:
                self.traj_type = traj_type
        if "openMM" in split_str:
            self.open_MM = True
        if FULL_RUN_NAME in name:
            self.is_real_run = True
        # get the remainder of the string
        if self.central_molecule:
            split_str.remove(self.central_molecule)
        if self.rotating_molecule:
            split_str.remove(self.rotating_molecule)
        if self.grid_type:
            split_str.remove(self.grid_type)
        if self.num_grid_points:
            split_str.remove(str(self.num_grid_points))
        if self.traj_type:
            split_str.remove(self.traj_type)
        if not self.ordering:
            split_str.remove("NO")
        if self.is_real_run:
            split_str.remove(FULL_RUN_NAME)
        if self.open_MM:
            split_str.remove("openMM")
        self.additional_data = "_".join(split_str)

    def _read_dict(self, dict_name):
        # TODO: it must be possible to refactor dict parsing for NameParser
        self.central_molecule = dict_name.pop("central_molecule", None)
        self.rotating_molecule = dict_name.pop("rotating_molecule", None)
        self.t_grid = dict_name.pop("t_grid", None)
        self.o_grid = dict_name.pop("o_grid", None)
        self.b_grid = dict_name.pop("b_grid", None)
        self.grid_type = dict_name.pop("grid_type", None)
        self.ordering = dict_name.pop("ordering", True)
        self.num_grid_points = dict_name.pop("num_grid_points", None)
        self.traj_type = dict_name.pop("traj_type", None)
        self.open_MM = dict_name.pop("open_MM", False)
        self.is_real_run = dict_name.pop("is_real_run", False)
        self.additional_data = dict_name.pop("additional_data", None)
        self.ending = dict_name.pop("ending", None)

    def get_dict_properties(self):
        return vars(self)

    def get_standard_name(self):
        standard_name = ""
        if self.central_molecule:
            standard_name += self.central_molecule + "_"
        if self.rotating_molecule:
            standard_name += self.rotating_molecule + "_"
        if self.is_real_run:
            standard_name += FULL_RUN_NAME + "_"
        if self.grid_type:
            standard_name += self.grid_type + "_"
        if self.t_grid and self.o_grid and self.b_grid:
            standard_name += f"o_{self.o_grid}_"
            standard_name += f"b_{self.b_grid}_"
            standard_name += f"t_{self.t_grid}_"
        if not self.ordering:
            standard_name += "NO_"
        if self.num_grid_points:
            standard_name += str(self.num_grid_points) + "_"
        if self.traj_type:
            standard_name += self.traj_type + "_"
        if self.open_MM:
            standard_name += "openMM_"
        if standard_name.endswith("_"):
            standard_name = standard_name[:-1]
        return standard_name

    def get_human_readable_name(self):
        # TODO: name eg H2O-H2O system, icosahedron grid, 22 rotations
        pass

    def get_grid_type(self):
        if not self.grid_type:
            raise ValueError(f"No grid type given!")
        return self.grid_type

    def get_traj_type(self):
        if not self.traj_type:
            raise ValueError(f"No traj type given!")
        return self.traj_type

    def get_num(self):
        if not self.num_grid_points:
            raise ValueError(f"No number given!")
        return self.num_grid_points


def particle_type2element(particle_type: str) -> str:
    """
    A helper function to convert gromacs particle type to element name readable by mendeleev.

    Args:
        particle_type: text written at characters 10:15 in a standard GROMACS line.

    Returns:
        element name, one of the names in periodic system (or ValueError)
    """
    ptable = fetch_table('elements')
    all_symbols = ptable["symbol"]
    # option 1: atom_name written in gro file is equal to element name (N, Na, Cl ...) -> directly use as element
    if particle_type in all_symbols.values:
        element_name = particle_type
    # option 2 (special case): CA is a symbol of alpha-carbon
    elif particle_type.startswith("CA"):
        element_name = "C"
    # option 3: first two letters are the name of a typical ion in upper case
    elif particle_type[:2] in ["CL", "MG", "RB", "CS", "LI", "ZN", "NA"]:
        element_name = particle_type.capitalize()[:2]
    # option 4: special case for calcium = C0
    elif particle_type[:2] == "C0":
        element_name = "Ca"
    # option 5: first letter is the name of the element in upper case
    elif particle_type[0] in all_symbols.values:
        element_name = particle_type[0]
    # error if still unable to determine the element
    else:
        message = f"I do not know how to extract element name from GROMACS atom type {particle_type}."
        raise ValueError(message)
    return element_name


class ParsedMolecule:

    def __init__(self, atoms: AtomGroup, box = None):
        self.atoms = atoms
        self.box = box
        self._update_attributes()

    def _update_attributes(self):
        self.num_atoms = len(self.atoms)
        self.atom_labels = self.atoms.names
        self.atom_types = self.atoms.types

    def get_atoms(self) -> AtomGroup:
        return self.atoms

    def get_center_of_mass(self) -> NDArray:
        return self.atoms.center_of_mass()

    def get_positions(self) -> NDArray:
        return self.atoms.positions

    def get_box(self):
        return self.box

    def __str__(self):
        return f"<ParsedMolecule with atoms {self.atom_types} at {self.get_center_of_mass()}>"

    def _rotate(self, about_what: str, rotational_obj: Rotation, inverse: bool = False):
        if about_what == "origin":
            point = (0, 0, 0)
        elif about_what == "body":
            point = self.get_center_of_mass()
        else:
            raise ValueError(f"Rotation about {about_what} unknown, try 'origin' or 'body'.")
        if inverse:
            rotational_obj = rotational_obj.inv()
        self.atoms.rotate(rotational_obj.as_matrix(), point=point)

    def rotate_about_origin(self, rotational_obj: Rotation, inverse: bool = False):
        self._rotate(about_what="origin", rotational_obj=rotational_obj, inverse=inverse)

    def rotate_about_body(self, rotational_obj: Rotation, inverse: bool = False):
        self._rotate(about_what="body", rotational_obj=rotational_obj, inverse=inverse)

    def translate(self, vector: np.ndarray):
        self.atoms.translate(vector)

    def translate_to_origin(self):
        current_position = self.get_center_of_mass()
        self.translate(-current_position)

    def translate_radially(self, distance_change: float):
        """
        Moves the object away from the origin in radial direction for the amount specified by distance_change (or
        towards the origin if a negative distance_change is given). If the object is at origin, translate in z-direction
        of the internal coordinate system.

        Args:
            distance_change: the change in length of the vector origin-object
        """
        # need to work with rounding because gromacs files only have 3-point precision
        initial_vector = np.round(self.get_center_of_mass(), 3)
        if np.allclose(initial_vector, [0, 0, 0], atol=1e-3):
            initial_vector = np.array([0, 0, 1])
        len_initial = np.linalg.norm(initial_vector)
        rescaled_vector = distance_change*initial_vector/len_initial
        self.atoms.translate(rescaled_vector)


class FileParser:

    def __init__(self, path_topology: str, path_trajectory: str = None):
        """
        This module serves to load a structure or trajectory information and output it in a standard format of
        a ParsedMolecule or a generator of ParsedMolecules (one per frame). No properties should be accessed directly,
        all work in other modules should be based on attributes and methods of ParsedMolecule.

        Args:
            path: a full path to the structure or trajectory file (.gro, .xtc, .xyz ....) that should be read
        """
        self.path_topology = path_topology
        self.path_trajectory = path_trajectory
        self.path_topology, self.path_trajectory = self.try_to_add_extension()
        if path_trajectory is not None:
            self.universe = mda.Universe(path_topology, path_trajectory)
        else:
            self.universe = mda.Universe(self.path_topology)

    def try_to_add_extension(self) -> List[str]:
        possible_exts = ['PSF', 'TOP', 'PRMTOP', 'PARM7', 'PDB', 'ENT', 'XPDB', 'PQR', 'GRO', 'CRD', 'PDBQT', 'DMS',
                         'TPR', 'MOL2', 'DATA', 'LAMMPSDUMP', 'XYZ', 'TXYZ', 'ARC', 'GMS', 'CONFIG', 'HISTORY', 'XML',
                         'MMTF', 'GSD', 'MINIMAL', 'ITP', 'IN', 'FHIAIMS', 'PARMED', 'RDKIT', 'OPENMMTOPOLOGY',
                         'OPENMMAPP']
        to_return = [self.path_topology, self.path_trajectory]
        for i, path in enumerate(to_return):
            if path is None:
                continue
            root, ext = os.path.splitext(path)
            if not ext:
                options = []
                for possible_ext in possible_exts:
                    test_file = f"{path}.{possible_ext.lower()}"
                    if os.path.isfile(test_file):
                        options.append(test_file)
                # if no files found, complain
                if len(options) == 0:
                    raise FileNotFoundError(f"No file with name {path} could be found and adding standard "
                                            f"extensions did not help.")
                # if exactly one file found, use it
                elif len(options) == 1:
                    to_return[i] = options[0]
                # if several files with same name but different extensions found, complain
                else:
                    raise AttributeError(f"More than one file corresponds to the name {path}. "
                                         f"Possible choices: {options}. "
                                         f"Please specify the extension you want to use.")
        return to_return

    def get_file_path_trajectory(self):
        return self.path_trajectory

    def get_file_path_topology(self):
        return self.path_topology

    def get_file_name(self):
        return Path(self.path_topology).stem

    def as_parsed_molecule(self) -> ParsedMolecule:
        return ParsedMolecule(self.universe.atoms, box=self.universe.dimensions)

    def generate_frame_as_molecule(self) -> Generator[ParsedMolecule, None, None]:
        for _ in self.universe.trajectory:
            yield ParsedMolecule(self.universe.atoms, box=self.universe.dimensions)


class PtParser(FileParser):

    def __init__(self, m1_path: str, m2_path: str, path_topology: str, path_trajectory: str = None):
        super().__init__(path_topology, path_trajectory)
        self.c_num = FileParser(m1_path).as_parsed_molecule().num_atoms
        self.r_num = FileParser(m2_path).as_parsed_molecule().num_atoms
        assert self.c_num + self.r_num == self.as_parsed_molecule().num_atoms

    def generate_frame_as_molecule(self) -> Generator[Tuple[ParsedMolecule, ParsedMolecule], None, None]:
        for _ in self.universe.trajectory:
            yield ParsedMolecule(self.universe.atoms[:self.c_num], box=self.universe.dimensions), ParsedMolecule(self.universe.atoms[self.c_num:], box=self.universe.dimensions)


class TranslationParser(object):

    def __init__(self, user_input: str):
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
        # we use a (shortened) hash value to uniquely identify the grid used, no matter how it's generated
        self.grid_hash = int(hashlib.md5(self.trans_grid).hexdigest()[:8], 16)
        # save the grid (for record purposes)
        path = f"{PATH_OUTPUT_TRANSGRIDS}trans_{self.grid_hash}.txt"
        # noinspection PyTypeChecker
        np.savetxt(path, self.trans_grid)

    def get_trans_grid(self) -> np.ndarray:
        return self.trans_grid

    def get_N_trans(self) -> int:
        return len(self.trans_grid)

    def sum_increments_from_first_radius(self):
        return np.sum(self.get_increments()[1:])

    def get_increments(self):
        increment_grid = [self.trans_grid[0]]
        for start, stop in zip(self.trans_grid, self.trans_grid[1:]):
            increment_grid.append(stop-start)
        increment_grid = np.array(increment_grid)
        assert np.all(increment_grid > 0), "Negative or zero increments in translation grid make no sense!"
        return increment_grid

    def _read_within_brackets(self) -> tuple:
        str_in_brackets = self.user_input.split('(', 1)[1].split(')')[0]
        str_in_brackets = literal_eval(str_in_brackets)
        if isinstance(str_in_brackets,numbers.Number):
            str_in_brackets = tuple((str_in_brackets,))
        return str_in_brackets
