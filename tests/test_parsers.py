import numpy as np
from mendeleev import element

from molgri.parsers import NameParser, BaseGroParser, TranslationParser


def test_atom_gro_file():
    file_name = f"molgri/examples/NA.gro"
    my_parser = BaseGroParser(file_name)
    assert my_parser.num_atoms == 1
    assert my_parser.comment == "Na+ ion"
    assert np.allclose(my_parser.box, [30, 30, 30])
    assert isinstance(my_parser.box[0], int)
    my_molecule = my_parser.molecule_set
    assert np.allclose(my_molecule.position, [0, 0, 0])
    assert my_molecule.atoms[0].element == element("Na")


def test_water_gro_file():
    file_name = f"molgri/examples/H2O.gro"
    my_parser = BaseGroParser(file_name)
    assert my_parser.num_atoms == 3
    assert my_parser.comment == "Water"
    assert np.allclose(my_parser.box, [30, 30, 30])
    my_molecule = my_parser.molecule_set
    # the atomic positions
    assert np.allclose(my_molecule.atoms[0].position, [0.000, -0.005, 0.004])
    assert np.allclose(my_molecule.atoms[1].position, [0.000,  -0.005,  -0.092])
    assert np.allclose(my_molecule.atoms[2].position, [0.000,   0.087,   0.028])
    # test elements
    assert my_molecule.atoms[0].element == element("O")
    assert my_molecule.atoms[1].element == element("H")
    assert my_molecule.atoms[1].element == element("H")
    # test gro labels
    assert my_molecule.atoms[0].gro_label == "OW"
    assert my_molecule.atoms[1].gro_label == "HW1"
    assert my_molecule.atoms[2].gro_label == "HW2"


def test_protein_gro_file():
    """
    Example in which more than one time step is accidentally provided to the parser. Intended behaviour: read the
    first time step, ignore all subsequent ones.
    """
    file_name = f"molgri/examples/example_protein.gro"
    my_parser = BaseGroParser(file_name)
    assert my_parser.num_atoms == 902
    assert my_parser.comment == "Protein in water t=   0.00000 step= 0"
    assert np.allclose(my_parser.box, [6.38830,  6.16418,   8.18519])
    assert isinstance(my_parser.box[0], float)
    my_molecule = my_parser.molecule_set
    assert np.allclose(my_molecule.atoms[0].position, [-0.421,  -0.191,  -1.942])
    assert np.allclose(my_molecule.atoms[1].position, [-0.450,  -0.287,  -1.946])
    assert np.allclose(my_molecule.atoms[901].position, [0.065,  -0.214,   2.135])
    assert my_molecule.atoms[0].element == element("N")
    assert my_molecule.atoms[0].gro_label == "N"
    assert my_molecule.atoms[1].element == element("H")
    assert my_molecule.atoms[1].gro_label == "H1"
    assert my_molecule.atoms[4].element == element("C")
    assert my_molecule.atoms[4].gro_label == "CA"
    assert my_molecule.atoms[5].element == element("H")
    assert my_molecule.atoms[5].gro_label == "HA"
    assert my_molecule.atoms[345].element == element("C")
    assert my_molecule.atoms[345].gro_label == "CA"
    assert my_molecule.atoms[901].element == element("O")
    assert my_molecule.atoms[901].gro_label == "OC2"
    all_elements = [at.element.symbol for at in my_molecule.atoms]
    organic_elements = ["N", "O", "C", "H", "S", "P", "F"]
    for el in all_elements:
        assert el in organic_elements, f"{el} not correctly identified as organic element"


def test_name_parser():
    example_names = ["protein0_CL_ico_30_full", "H2O_H2O_ico_4_circular", "CL_NA_full_ico_5"]
    np1 = NameParser(example_names[0])
    np2 = NameParser(example_names[1])
    np3 = NameParser(example_names[2])
    assert np1.get_standard_name() == example_names[0]
    assert np2.get_standard_name() == example_names[1]
    assert np3.get_standard_name() == "CL_NA_ico_5_full"


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


if __name__ == '__main__':
    test_atom_gro_file()
    test_water_gro_file()
    test_protein_gro_file()
    test_name_parser()
    test_trans_parser()