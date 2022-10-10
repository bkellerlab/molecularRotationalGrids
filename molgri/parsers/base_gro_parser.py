from mendeleev.fetch import fetch_table
import numpy as np

from ..objects.molecule import Molecule


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
        self.box = tuple([float(x) for x in self.gro_file.readline().strip().split()])
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


if __name__ == '__main__':
    from molgri.paths import PATH_INPUT_BASEGRO
    particle_type2element("C0")
    bgp = BaseGroParser(gro_read=f"{PATH_INPUT_BASEGRO}H2O.gro", parse_atoms=True)
    print(bgp.molecule_set.atoms[2].element)
    print(bgp.molecule_set.atoms[2].gro_label)
