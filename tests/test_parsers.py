import numpy as np
from mendeleev import element

from molecularRotationalGrids.parsers.base_gro_parser import BaseGroParser


def test_atom_gro_file():
    file_name = f"tests/example_NA.gro"
    my_parser = BaseGroParser(file_name)
    assert my_parser.num_atoms == 1
    assert my_parser.comment == "Na+ ion"
    assert np.allclose(my_parser.box, [30, 30, 30])
    my_molecule = my_parser.molecule_set
    assert np.allclose(my_molecule.position, [0, 0, 0])
    assert my_molecule.atoms[0].element == element("Na")


if __name__ == '__main__':
    test_atom_gro_file()