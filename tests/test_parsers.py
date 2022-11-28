import numpy as np

from molgri.parsers import TranslationParser, FileParser
from molgri.constants import ANGSTROM2NM


def test_atom_gro_file():
    file_name = f"molgri/examples/NA.gro"
    my_parser = FileParser(file_name).as_parsed_molecule()
    assert my_parser.num_atoms == 1
    assert np.allclose(my_parser.box[:3]*ANGSTROM2NM, [30, 30, 30])
    my_molecule_set = my_parser
    my_molecule = my_molecule_set
    assert np.allclose(my_molecule.get_center_of_mass(), [0, 0, 0])
    assert my_molecule.atom_labels[0] == "NA"
    assert np.isclose(my_molecule.atoms.masses, [22.98977])


def test_water_gro_file():
    file_name = f"molgri/examples/H2O.gro"
    my_parser = FileParser(file_name).as_parsed_molecule()
    assert my_parser.num_atoms == 3
    assert np.allclose(my_parser.box[:3]*ANGSTROM2NM, [30, 30, 30])
    my_molecule = my_parser
    # the atomic get_positions()
    assert np.allclose(my_molecule.atoms[0].position*ANGSTROM2NM, [0.000, -0.005, 0.004])
    assert np.allclose(my_molecule.atoms[1].position*ANGSTROM2NM, [0.000,  -0.005,  -0.092])
    assert np.allclose(my_molecule.atoms[2].position*ANGSTROM2NM, [0.000,   0.087,   0.028])
    # test elements
    assert my_molecule.atom_types[0] == "O"
    assert my_molecule.atom_types[1] == "H"
    assert my_molecule.atom_types[2] == "H"
    # test gro labels
    assert my_molecule.atom_labels[0] == "OW"
    assert my_molecule.atom_labels[1] == "HW1"
    assert my_molecule.atom_labels[2] == "HW2"


def test_protein_gro_file():
    """
    Example in which more than one time step is accidentally provided to the parser. Intended behaviour: read the
    first time step, ignore all subsequent ones.
    """
    file_name = f"molgri/examples/example_protein.gro"
    my_parser = FileParser(file_name).as_parsed_molecule()
    assert my_parser.num_atoms == 902
    assert np.allclose(my_parser.box[:3]*ANGSTROM2NM, [6.38830,  6.16418,   8.18519])
    my_molecule = my_parser
    assert np.allclose(my_molecule.atoms[0].position*ANGSTROM2NM, [-0.421,  -0.191,  -1.942])
    assert np.allclose(my_molecule.atoms[1].position*ANGSTROM2NM, [-0.450,  -0.287,  -1.946])
    assert np.allclose(my_molecule.atoms[901].position*ANGSTROM2NM, [0.065,  -0.214,   2.135])
    assert my_molecule.atom_labels[0] == "N"
    assert my_molecule.atom_types[0] == "N"
    assert my_molecule.atom_labels[1] == "H1"
    assert my_molecule.atom_types[1] == "H"
    assert my_molecule.atom_labels[4] == "CA"
    assert my_molecule.atom_types[4] == "C"
    assert my_molecule.atom_labels[5] == "HA"
    assert my_molecule.atom_types[5] == "H"
    assert my_molecule.atom_labels[345] == "CA"
    assert my_molecule.atom_types[345] == "C"
    assert my_molecule.atom_labels[901] == "OC2"
    assert my_molecule.atom_types[901] == "O"
    organic_elements = ["N", "O", "C", "H", "S", "P", "F"]
    for el in my_molecule.atom_types:
        assert el in organic_elements, f"{el} not correctly identified as organic element"


def test_parsing_xyz():
    file_name = "molgri/examples/glucose.xyz"
    my_parser = FileParser(file_name)
    my_molecule = my_parser.as_parsed_molecule()
    all_types = ["C"]*6
    all_types.extend(["O"]*6)
    assert np.all(my_molecule.atom_types == all_types)
    assert np.allclose(my_molecule.get_positions()[0], [35.884,  30.895,  49.120])
    assert np.allclose(my_molecule.get_positions()[1], [36.177,  29.853,  50.124])
    assert np.allclose(my_molecule.get_positions()[6], [34.968,  30.340,  48.234])
    assert np.allclose(my_molecule.get_positions()[11], [39.261,  32.018,  46.920])


def test_parsing_pdb():
    file_name = "molgri/examples/example_pdb.pdb"
    my_parser = FileParser(file_name)
    my_molecule = my_parser.as_parsed_molecule()
    assert np.all(my_molecule.atom_types[0] == "N")
    assert np.all(my_molecule.atom_types[1] == "C")
    assert np.all(my_molecule.atom_types[10] == "H")
    assert np.all(my_molecule.atom_types[60] == "H")
    assert np.allclose(my_molecule.get_positions()[0], [0.000,   0.000,   0.000])
    assert np.allclose(my_molecule.get_positions()[1], [1.456,   0.000,   0.000])
    assert np.allclose(my_molecule.get_positions()[6], [2.010,   1.208,  -0.746])
    assert np.allclose(my_molecule.get_positions()[11], [3.241,  -0.000,   1.742])
    assert np.allclose(my_molecule.get_positions()[60], [14.680,  -0.233,   5.690])


def test_trans_parser():
    assert np.allclose(TranslationParser("1").get_trans_grid(), np.array([1]))
    assert np.allclose(TranslationParser("2.14").get_trans_grid(), np.array([2.14]))
    assert np.allclose(TranslationParser("(2, 3.5, 4)").get_trans_grid(), np.array([2, 3.5, 4]))
    assert np.allclose(TranslationParser("[16, 2, 11]").get_trans_grid(), np.array([2, 11, 16]))
    assert np.allclose(TranslationParser("1, 2.7, 2.6").get_trans_grid(), np.array([1, 2.6, 2.7]))
    assert np.allclose(TranslationParser("linspace(2, 6, 5)").get_trans_grid(), np.linspace(2, 6, 5))
    assert np.allclose(TranslationParser("linspace (2, 6, 5)").get_trans_grid(), np.linspace(2, 6, 5))
    assert np.allclose(TranslationParser("linspace(2, 6)").get_trans_grid(), np.linspace(2, 6, 50))
    assert np.allclose(TranslationParser("range(2, 7, 2)").get_trans_grid(), np.arange(2, 7, 2))
    assert np.allclose(TranslationParser("range(2, 7)").get_trans_grid(), np.arange(2, 7))
    assert np.allclose(TranslationParser("arange(2, 7)").get_trans_grid(), np.arange(2, 7))
    assert np.allclose(TranslationParser("range(7)").get_trans_grid(), np.arange(7))
    assert np.allclose(TranslationParser("range(7.3)").get_trans_grid(), np.arange(7.3))
    assert np.allclose(TranslationParser("arange(7.3)").get_trans_grid(), np.arange(7.3))
    # increments
    assert np.allclose(TranslationParser("[5, 6.5, 7, 12]").get_increments(), np.array([5, 1.5, 0.5, 5]))
    assert np.allclose(TranslationParser("[5, 12, 6.5, 7]").get_increments(), np.array([5, 1.5, 0.5, 5]))
    assert np.allclose(TranslationParser("[12, 5, 6.5, 7]").get_increments(), np.array([5, 1.5, 0.5, 5]))
    # sum of increments
    assert np.allclose(TranslationParser("[2, 2.5, 2.77, 3.4]").sum_increments_from_first_radius(), 1.4)
    # hash
    tp = TranslationParser("[2, 2.5, 2.77, 3.4]")
    assert tp.grid_hash == 1772331579
    tp2 = TranslationParser("range(2, 7, 2)")
    assert tp2.grid_hash == 481270436
    tp3 = TranslationParser("[2, 4, 6]")
    assert tp3.grid_hash == 481270436
    tp4 = TranslationParser("linspace(2, 6, 3)")
    assert tp4.grid_hash == 481270436


if __name__ == '__main__':
    test_atom_gro_file()
    test_water_gro_file()
    test_protein_gro_file()
    test_parsing_xyz()
    test_parsing_pdb()
    #test_name_parser()
    #test_trans_parser()
    #test_expected_errors()
