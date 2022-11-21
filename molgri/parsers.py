import hashlib
import numbers
import os
from typing import TextIO

import numpy as np
from mendeleev.fetch import fetch_table
from ast import literal_eval

from .bodies import Molecule, MoleculeSet
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


class BaseGroParser:

    def __init__(self, gro_read: str, parse_atoms: bool = True, gro_file_read: TextIO = None):
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
            gro_file_read: (optional) if gro file already opened, it can be provided here, in this case, gro_read
                            argument ignored

        """
        head, tail = os.path.split(gro_read)
        self.molecule_name = tail.split(".")[0]
        if not gro_file_read:
            self.gro_file = open(gro_read, "r")
        else:
            self.gro_file = gro_file_read
        self.comment = self.gro_file.readline().strip()
        if "t=" in self.comment:
            split_comment = self.comment.split("=")
            _t = split_comment[-1].strip()
            self.t = float(_t)
        else:
            self.t = None
        self.num_atoms = int(self.gro_file.readline().strip())
        self.atom_lines_nm = []
        for line in range(self.num_atoms):
            # Append exact copy of the current line in self.gro_file to self.atom_lines_nm (including \n at the end).
            line = self.gro_file.readline()
            self.atom_lines_nm.append(line)
        if parse_atoms:
            self.molecule_set = self._create_molecule(*self._parse_atoms())
        else:
            self.molecule_set = None
        self.box = tuple([literal_eval(x) for x in self.gro_file.readline().strip().split()])
        assert len(self.atom_lines_nm) == self.num_atoms
        #self.gro_file.close()

    def _parse_atoms(self) -> tuple:
        list_residues = []
        list_gro_labels = []
        list_atom_names = []
        list_atom_pos = []
        for line in self.atom_lines_nm:
            # read out each individual part of the atom position line
            residue_num = int(line[0:5])
            # residue_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            element_name = particle_type2element(atom_name)
            # atom_num = int(line[15:20])
            x_pos_nm = float(line[20:28])
            y_pos_nm = float(line[28:36])
            z_pos_nm = float(line[36:44])
            # optionally velocities in nm/ps are writen at characters 44:52, 52:60, 60:68 of the line
            list_residues.append(residue_num)
            list_gro_labels.append(atom_name)
            list_atom_names.append(element_name)
            list_atom_pos.append([x_pos_nm, y_pos_nm, z_pos_nm])
        return list_gro_labels, list_atom_names, list_atom_pos

    def _create_molecule(self, list_gro_labels, list_atom_names, list_atom_pos) -> Molecule:
        array_atom_pos = np.array(list_atom_pos)
        return Molecule(atom_names=list_atom_names, centers=array_atom_pos, center_at_origin=False,
                        gro_labels=list_gro_labels)


class PtFrameParser(BaseGroParser):

    def __init__(self, gro_read: str, parse_atoms: bool = True, gro_file_read: TextIO = None):
        # don't parse atoms in super() since you don't know yet which atoms belong to molecule 1 and which to 2.
        super().__init__(gro_read, parse_atoms=False, gro_file_read=gro_file_read)
        if "c_num=" in self.comment and "r_num=" in self.comment:
            split_comment = self.comment.split("=")
            _c_num = split_comment[1].split(",")[0]
            self.c_num = int(_c_num)
            _r_num = split_comment[2].split(",")[0]
            self.r_num = int(_r_num)
        else:
            raise ValueError(f"Cannot find c_num and/or r_nu in comment line: {self.comment}")
        if parse_atoms:
            self.molecule_set = self._create_molecule_set(*self._parse_atoms())

    def _create_molecule_set(self, list_gro_labels, list_atom_names, list_atom_pos) -> MoleculeSet:
        """
        Redefine this function so that the molecular set consists of exactly two molecules, one fixed, one rotated.
        """
        array_atom_pos = np.array(list_atom_pos)
        molecule_list = [Molecule(atom_names=list_atom_names[0:self.c_num],
                                  centers=array_atom_pos[0:self.c_num], center_at_origin=False,
                                  gro_labels=list_gro_labels[0:self.c_num])]
        assert self.c_num+self.r_num == self.num_atoms
        molecule_list.append(Molecule(atom_names=list_atom_names[self.c_num:],
                                      centers=array_atom_pos[self.c_num:], center_at_origin=False,
                                      gro_labels=list_gro_labels[self.c_num:]))
        return MoleculeSet(molecule_list)


class MultiframeGroParser:

    def __init__(self, gro_read: str, parse_atoms: bool = True, is_pt=True):
        self.timesteps = []
        with open(gro_read) as f:
            while True:
                try:
                    if is_pt:
                        bgp = PtFrameParser(gro_read, parse_atoms=parse_atoms, gro_file_read=f)
                    else:
                        bgp = BaseGroParser(gro_read, parse_atoms=parse_atoms, gro_file_read=f)
                except ValueError:
                    # this occurs at the end of the file when no more timesteps to read.
                    break
                self.timesteps.append(bgp)


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
