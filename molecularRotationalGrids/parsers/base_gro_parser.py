from typing import TextIO
from mendeleev import element
import numpy as np
from objects_3d.molecule import Atom, Molecule
#from objects_3d.shape_set import MoleculeSet


class BaseGroParser:

    def __init__(self, gro_read: str, parse_atoms=True, gro_write: TextIO = None):
        """
        This parser is used for gro files with only one time point.

        Args:
            gro_file: a file to be parsed (already opened)
        """
        gro_file = open(gro_read, "r")
        self.f_write = gro_write
        self.comment = gro_file.readline().strip()
        self.num_atoms = int(gro_file.readline().strip())
        self.atom_lines = []
        self.properties_per_line = []
        self.molecule_set = None
        for line in range(self.num_atoms):
            self._parse_line(gro_file)
        if parse_atoms:
            self._parse_atoms()
        self.box = tuple(gro_file.readline().strip().split())
        assert len(self.atom_lines) == self.num_atoms
        gro_file.close()

    def _parse_line(self, gro_file):
        line = gro_file.readline()
        self.atom_lines.append(line)
        dict_properties = dict()
        dict_properties["residue_num"] = int(line[0:5])
        dict_properties["residue_name"] = line[5:10].strip()
        dict_properties["atom_name"] = line[10:15].strip()

        try:
            dict_properties["atom_symbol"] = element(dict_properties["atom_name"][0]).symbol
            if dict_properties["atom_name"] == "CL" or dict_properties["atom_name"] == "NA":
                dict_properties["atom_symbol"] = element(dict_properties["atom_name"].capitalize()).symbol
        except:
            dict_properties["atom_symbol"] = element(dict_properties["atom_name"][0:1]).symbol
        dict_properties["atom_num"] = int(line[15:20])
        # the units of distances are nm
        dict_properties["x_pos"] = float(line[20:28])
        dict_properties["y_pos"] = float(line[28:36])
        dict_properties["z_pos"] = float(line[36:44])
        try:
            dict_properties["x_vel"] = float(line[44:52])
            dict_properties["y_vel"] = float(line[52:60])
            dict_properties["z_vel"] = float(line[60:68])
        except ValueError:
            # velocities not always written
            pass
        self.properties_per_line.append(dict_properties)

    def _parse_atoms(self):
        #atoms = []
        list_symbols = []
        list_pos = []
        for line_dict in self.properties_per_line:
            symbol = line_dict["atom_symbol"]
            position = line_dict["x_pos"], line_dict["y_pos"], line_dict["z_pos"]
            list_symbols.append(symbol)
            list_pos.append(position)
            #my_atom = Atom(symbol, start_position=np.array(position))
            #atoms.append(my_atom)
        self.molecule_set = Molecule(atom_names=list_symbols, centers=list_pos)



if __name__ == '__main__':
    from my_constants import *
    file_path = f"{PATH_BASE_GRO_FILES}CL.gro"
    my_parser = BaseGroParser(file_path)