import re
import numbers

import numpy as np
from mendeleev.fetch import fetch_table
from ast import literal_eval

from .bodies import Molecule
from .constants import MOLECULE_NAMES, SIX_METHOD_NAMES, FULL_RUN_NAME


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
        self.central_molecule = dict_name.pop("central_molecule", None)
        self.rotating_molecule = dict_name.pop("rotating_molecule", None)
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


class BaseGroParser:

    def __init__(self, gro_read: str, parse_atoms: bool = True):
        """
        This parser reads the data from a .gro file. If multiple time steps are written, it only reads the first one.

        If you want to access or copy parts of the .gro file (comment, number of atoms, atom position lines, box)
        to another file, select parse_atoms=False (this is faster) and use the data saved in self.comment,
        self.num_atoms, self.atom_lines_nm and self.box.

        If you want to read the .gro file in order to translate/rotate atoms in it, select parse_atoms=True and
        access the Molecule object under self.molecule_set.

        Args:
            gro_read: the path to the .gro file to be parsed
            parse_atoms: select True if you want to manipulate (rotate/translate) any atoms from this .gro file;
                         select False if you only want to copy the atom position lines as-provided
        """
        self.gro_file = open(gro_read, "r")
        self.comment = self.gro_file.readline().strip()
        self.num_atoms = int(self.gro_file.readline().strip())
        self.atom_lines_nm = []
        for line in range(self.num_atoms):
            # Append exact copy of the current line in self.gro_file to self.atom_lines_nm (including \n at the end).
            line = self.gro_file.readline()
            self.atom_lines_nm.append(line)
        if parse_atoms:
            a_labels, a_names, a_pos = self._parse_atoms()
            a_pos = np.array(a_pos)
            self.molecule_set = Molecule(atom_names=a_names, centers=a_pos, center_at_origin=False,
                                         gro_labels=a_labels)
        else:
            self.molecule_set = None
        self.box = tuple([literal_eval(x) for x in self.gro_file.readline().strip().split()])
        assert len(self.atom_lines_nm) == self.num_atoms
        self.gro_file.close()

    def _parse_atoms(self) -> tuple:
        list_gro_labels = []
        list_atom_names = []
        list_atom_pos = []
        for line in self.atom_lines_nm:
            # read out each individual part of the atom position line
            # residue_num = int(line[0:5])
            # residue_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            element_name = particle_type2element(atom_name)
            # atom_num = int(line[15:20])
            x_pos_nm = float(line[20:28])
            y_pos_nm = float(line[28:36])
            z_pos_nm = float(line[36:44])
            # optionally velocities in nm/ps are writen at characters 44:52, 52:60, 60:68 of the line
            list_gro_labels.append(atom_name)
            list_atom_names.append(element_name)
            list_atom_pos.append([x_pos_nm, y_pos_nm, z_pos_nm])
        return list_gro_labels, list_atom_names, list_atom_pos


class TranslationParser(object):

    def __init__(self, user_input: str):
        self.user_input = user_input
        if "linspace" in self.user_input:
            bracket_input = self._read_within_brackets()
            self.trans_grid = np.linspace(*bracket_input)
        elif "range" in self.user_input:
            bracket_input = self._read_within_brackets()
            self.trans_grid = np.arange(*bracket_input)
        else:
            self.trans_grid = literal_eval(self.user_input)
            self.trans_grid = np.array(self.trans_grid)
            self.trans_grid = np.sort(self.trans_grid, axis=None)

    def get_trans_grid(self) -> np.ndarray:
        return self.trans_grid

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
