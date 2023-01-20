"""
The parsers module deals with input. This can be a trajectory or single-frame topology file (FileParser),
specifically a Pseudotrajectory file (PtParser) or a string describing a translational grid (TranslationParser).
The result of file parsing is immediately transformed into a ParsedMolecule or a sequence of ParsedMolecule
objects, which is a wrapper that other modules should access.
"""

import hashlib
import numbers
import os
from pathlib import Path
from typing import Generator, Tuple, List

import numpy as np
from ast import literal_eval

from MDAnalysis.auxiliary.XVG import XVGReader
from numpy.typing import NDArray, ArrayLike
from scipy.spatial.transform import Rotation
import MDAnalysis as mda
from MDAnalysis.core import AtomGroup

from molgri.constants import EXTENSIONS, NM2ANGSTROM, GRID_ALGORITHMS, DEFAULT_ALGORITHM_O, ZERO_ALGORITHM, \
    DEFAULT_ALGORITHM_B
from molgri.paths import PATH_OUTPUT_TRANSGRIDS
from molgri.space.rotations import two_vectors2rot


class NameParser:

    def __init__(self, name_string: str):
        self.name_string = name_string
        self.N = self._find_a_number()
        self.algo = self._find_algorithm()

    def _find_a_number(self) -> int or None:
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
            return None

    def _find_algorithm(self) -> str or None:
        split_string = self.name_string.split("_")
        candidates = []
        for fragment in split_string:
            if fragment in GRID_ALGORITHMS:
                candidates.append(fragment)
            elif fragment.lower() == "none":
                candidates.append(ZERO_ALGORITHM)
        # >= 2 algorithms found in the string
        if len(candidates) > 1:
            raise ValueError(f"Found two or more algorithm names in grid name {self.name_string}, can't decide.")
        # exactly one algorithm in the string -> return it
        elif len(candidates) == 1:
            return candidates[0]
        # no algorithm given -> None
        else:
            return None


class FullGridNameParser:

    def __init__(self, name_string: str):
        split_name = name_string.split("_")
        self.b_grid_name = None
        self.o_grid_name = None
        self.t_grid_name = None
        for i, split_part in enumerate(split_name):
            if split_part == "b":
                try:
                    self.b_grid_name = GridNameParser(
                        f"{split_name[i + 1]}_{split_name[i + 2]}", "b").get_standard_grid_name()
                except IndexError:
                    self.b_grid_name = GridNameParser(f"{split_name[i + 1]}", "o").get_standard_grid_name()
            elif split_part == "o":
                try:
                    self.o_grid_name = GridNameParser(
                        f"{split_name[i + 1]}_{split_name[i + 2]}").get_standard_grid_name()
                except IndexError:
                    self.o_grid_name = GridNameParser(f"{split_name[i + 1]}").get_standard_grid_name()
            elif split_part == "t":
                self.t_grid_name = f"{split_name[i + 1]}"

    def get_standard_full_grid_name(self):
        return f"o_{self.o_grid_name}_b_{self.b_grid_name}_t_{self.t_grid_name}"

    def get_num_b_rot(self):
        return int(self.b_grid_name.split("_")[1])

    def get_num_o_rot(self):
        return int(self.o_grid_name.split("_")[1])


class GridNameParser(NameParser):
    """
    Differently than pure NameParser, GridNameParser raises errors if the name doesn't correspond to a standard grid
    name.
    """
    # TODO: don't simply pass if incorrectly written alg name!
    def __init__(self, name_string: str, o_or_b="o"):
        super().__init__(name_string)
        # num of points 0 or 1 -> always zero algorithm; selected zero algorithm -> always num of points is 1
        if (self.N in (1, 0)) or (self.N is None and self.algo == ZERO_ALGORITHM):
            self.algo = ZERO_ALGORITHM
            self.N = 1
        # ERROR - zero algorithm but a larger number of points provided
        elif self.algo == ZERO_ALGORITHM and self.N > 1:
            raise ValueError(f"Zero algorithm selected but number of points {self.N}>1 in {self.name_string}.")
        # ERROR - no points but also not zero algorithm
        elif self.N is None and (self.algo != ZERO_ALGORITHM and self.algo is not None):
            raise ValueError(f"An algorithm provided ({self.algo}) but not number of points in {self.name_string}")
        # algorithm not provided but > 1 points -> default algorithm
        elif self.algo is None and self.N > 1 and o_or_b == "o":
            self.algo = DEFAULT_ALGORITHM_O
        elif self.algo is None and self.N > 1 and o_or_b == "b":
            self.algo = DEFAULT_ALGORITHM_B
        # ERROR - absolutely nothing provided
        elif self.algo is None and self.N is None:
            raise ValueError(f"Algorithm name and number of grid points not recognised in name {self.name_string}.")

    def get_standard_grid_name(self) -> str:
        return f"{self.algo}_{self.N}"


class ParsedMolecule:

    def __init__(self, atoms: AtomGroup, box=None):
        """
        Wraps the behaviour of AtomGroup and Box implemented by MDAnalysis and provides the necessary
        rotation and translation functions.

        Args:
            atoms: an AtomGroup that makes up the molecule
            box: the box in which the molecule is simulated
        """
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

    def rotate_to(self, position: NDArray):
        """
        1) rotate around origin to get to a rotational position described by position
        2) rescale radially to original length of position vector

        Args:
            position: 3D coordinates of a point on a sphere, end position of COM
        """
        assert len(position) == 3, "Position must be a 3D location in space"
        com_radius = np.linalg.norm(self.get_center_of_mass())
        position_radius = np.linalg.norm(position)
        rot_matrix = two_vectors2rot(self.get_center_of_mass(), position)
        self.rotate_about_origin(Rotation.from_matrix(rot_matrix))
        self.translate_radially(position_radius - com_radius)

    def translate_to_origin(self):
        """
        Translates the object so that the COM is at origin. Does not change the orientation.
        """
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
        This module serves to load topology or trajectory information and output it in a standard format of
        a ParsedMolecule or a generator of ParsedMolecules (one per frame). No properties should be accessed directly,
        all work in other modules should be based on attributes and methods of ParsedMolecule.

        Args:
            path_topology: a full path to the structure file (.gro, .xtc, .xyz ....) that should be read
            path_trajectory: a full path to the trajectory file (.xtc, .xyz ....) that should be read
        """
        self.path_topology = path_topology
        self.path_trajectory = path_trajectory
        self.path_topology, self.path_trajectory = self._try_to_add_extension()
        if path_trajectory is not None:
            self.universe = mda.Universe(path_topology, path_trajectory)
        else:
            self.universe = mda.Universe(self.path_topology)

    def _try_to_add_extension(self) -> List[str]:
        """
        If no extension is added, the program will try to add standard extensions. If no or multiple files with
        standard extensions are added, it will raise an error. It is always possible to explicitly state an extension.
        """
        possible_exts = EXTENSIONS
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
        """
        Method to use if you want to parse only a single molecule, eg. if only a topology file was provided.

        Returns:
            a parsed molecule from current or only frame
        """
        return ParsedMolecule(self.universe.atoms, box=self.universe.dimensions)

    def generate_frame_as_molecule(self) -> Generator[ParsedMolecule, None, None]:
        """
        Method to use if you want to parse an entire trajectory and generate a ParsedMolecule for each frame.

        Returns:
            a generator yielding each frame individually
        """
        for _ in self.universe.trajectory:
            yield ParsedMolecule(self.universe.atoms, box=self.universe.dimensions)


class PtParser(FileParser):

    def __init__(self, m1_path: str, m2_path: str, path_topology: str, path_trajectory: str = None):
        """
        A parser specifically for Pseudotrajectories written with the molgri.writers module. Useful to test the
        behaviour of writers or to analyse the resulting pseudo-trajectories.

        Args:
            m1_path: full path to the topology of the central molecule
            m2_path: full path to the topology of the rotating molecule
            path_topology: full path to the topology of the combined system
            path_trajectory: full path to the trajectory of the combined system
        """
        super().__init__(path_topology, path_trajectory)
        self.c_num = FileParser(m1_path).as_parsed_molecule().num_atoms
        self.r_num = FileParser(m2_path).as_parsed_molecule().num_atoms
        assert self.c_num + self.r_num == self.as_parsed_molecule().num_atoms

    def generate_frame_as_molecule(self) -> Generator[Tuple[ParsedMolecule, ParsedMolecule], None, None]:
        """
        Differently to FileParser, this generator always yields a pair of molecules: central, rotating (for each frame).

        Yields:
            (central molecule, rotating molecule) at each frame of PT

        """
        for _ in self.universe.trajectory:
            c_molecule = ParsedMolecule(self.universe.atoms[:self.c_num], box=self.universe.dimensions)
            r_molecule = ParsedMolecule(self.universe.atoms[self.c_num:], box=self.universe.dimensions)
            yield c_molecule, r_molecule


class TranslationParser(object):

    def __init__(self, user_input: str):
        """
        User input is expected in nanometers (nm)!

        Parse all ways in which the user may provide a linear translation grid. Currently supported formats:
            - a list of numbers, eg '[1, 2, 3]'
            - a linearly spaced list with optionally provided number of elements eg. 'linspace(1, 5, 50)'
            - a range with optionally provided step, eg 'range(0.5, 3, 0.4)'

        Args:
            user_input: a string in one of allowed formats
        """
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
        # convert to angstrom
        self.trans_grid = self.trans_grid * NM2ANGSTROM
        # we use a (shortened) hash value to uniquely identify the grid used, no matter how it's generated
        self.grid_hash = int(hashlib.md5(self.trans_grid).hexdigest()[:8], 16)
        # save the grid (for record purposes)
        path = f"{PATH_OUTPUT_TRANSGRIDS}trans_{self.grid_hash}.txt"
        # noinspection PyTypeChecker
        np.savetxt(path, self.trans_grid)

    def get_trans_grid(self) -> NDArray:
        return self.trans_grid

    def get_N_trans(self) -> int:
        return len(self.trans_grid)

    def sum_increments_from_first_radius(self) -> ArrayLike:
        """
        Get final distance - first non-zero distance == sum(increments except the first one).

        Useful because often the first radius is large and then only small increments are made.
        """
        return np.sum(self.get_increments()[1:])

    def get_increments(self) -> NDArray:
        """
        Get an array where each element represents an increment needed to get to the next radius.

        Example:
            self.trans_grid = np.array([10, 10.5, 11.2])
            self.get_increments() -> np.array([10, 0.5, 0.7])
        """
        increment_grid = [self.trans_grid[0]]
        for start, stop in zip(self.trans_grid, self.trans_grid[1:]):
            increment_grid.append(stop-start)
        increment_grid = np.array(increment_grid)
        assert np.all(increment_grid > 0), "Negative or zero increments in translation grid make no sense!"
        return increment_grid

    def _read_within_brackets(self) -> tuple:
        """
        Helper function to aid reading linspace(start, stop, num) and arange(start, stop, step) formats.
        """
        str_in_brackets = self.user_input.split('(', 1)[1].split(')')[0]
        str_in_brackets = literal_eval(str_in_brackets)
        if isinstance(str_in_brackets, numbers.Number):
            str_in_brackets = tuple((str_in_brackets,))
        return str_in_brackets


class XVGParser(object):

    def __init__(self, path_xvg: str):
        # this is done in order to function with .xvg ending or with no ending
        if not path_xvg.endswith(".xvg"):
            self.path_name = f"{path_xvg}.xvg"
        else:
            self.path_name = path_xvg
        reader = XVGReader(self.path_name)
        self.all_values = reader._auxdata_values
        reader.close()

    def get_all_columns(self) -> NDArray:
        return self.all_values

    def get_y_unit(self) -> str:
        y_unit = None
        with open(self.path_name, 'r') as f:
            for line in f:
                # parse property unit
                if line.startswith("@    yaxis  label"):
                    split_line = line.split('"')
                    y_unit = split_line[1]
                    break
        if y_unit is None:
            print("Warning: energy units could not be detected in the xvg file.")
            y_unit = "[?]"
        return y_unit

    def get_column_index_by_name(self, column_label) -> Tuple[str, int]:
        correct_column = None
        with open(self.path_name, 'r') as f:
            for line in f:
                # parse column number
                if f'"{column_label}"' in line:
                    split_line = line.split(" ")
                    correct_column = int(split_line[1][1:]) + 1
                if not line.startswith("@") and not line.startswith("#"):
                    break
        if correct_column is None:
            print(f"Warning: a column with label {column_label} not found in the XVG file. Using the first y-axis "
                  f"column instead.")
            column_label = "XVG column 1"
            correct_column = 1
        return column_label, correct_column
