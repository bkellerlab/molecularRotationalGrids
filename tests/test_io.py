import numpy as np

from molgri.io import EnergyReader
from molgri.paths import PATH_TEST_FILES


def test_energy_reader():
    my_e_reader = EnergyReader()
    # ONE COLUMN FILE #
    file1 = f"{PATH_TEST_FILES}one_column_xvg_file.xvg"
    one_column_df = my_e_reader.load_energy(file1)
    # correct length
    assert len(one_column_df) == 1001, f"one_column_xvg_file should contain 1001 elements not {len(one_column_df)}"
    # correct first element
    assert np.allclose(one_column_df.iloc[0], [0.000000,  0.899576])
    # correct last element
    assert np.allclose(one_column_df.iloc[-1], [1000.000000, 0.995087])
    # correct in-between element
    assert np.allclose(one_column_df.iloc[50], [50.000000,  0.866821])
    # correct titles
    assert len(one_column_df.columns) == 2
    assert np.all(one_column_df.columns == ["Time [ps]", "Pressure"])
    assert np.allclose(my_e_reader.load_single_energy_column(file1, "Pressure"), one_column_df["Pressure"])

    # MULTIPLE COLUMN FILE #
    file2 = f"{PATH_TEST_FILES}multiple_column_xvg_file.xvg"
    one_column_df = my_e_reader.load_energy(file2)
    # correct length
    assert len(one_column_df) == 1998, f"one_column_xvg_file should contain 1998 elements not {len(one_column_df)}"
    # correct first element
    assert np.allclose(one_column_df.iloc[0], [0.000000,   -0.039548,   -0.000008,  -20.538410,  -17.047235])
    # correct last element
    assert np.allclose(one_column_df.iloc[-1], [19.970000,   -0.012678,   -0.000009,   -1.976318,   69.296021])
    # correct in-between element
    assert np.allclose(one_column_df.iloc[50], [0.500000,   -0.362602,   -0.000008, -13.056232,   -8.251482])
    # correct titles
    assert len(one_column_df.columns) == 5
    assert np.all(one_column_df.columns == ['Time [ps]', 'LJ (SR)', 'Disper. corr.', 'Coulomb (SR)', 'Potential'])
    # specific values of specific columns
    assert np.isclose(my_e_reader.load_single_energy_column(file2, 'LJ (SR)')[0], -0.039548)
    assert np.isclose(my_e_reader.load_single_energy_column(file2, 'Coulomb (SR)')[-1], -1.976318)
    assert np.isclose(my_e_reader.load_single_energy_column(file2, 'Potential')[7], -24.412426)


# def test_trans_parser():
#     assert np.allclose(TranslationParser("1").get_trans_grid(), np.array([1])*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("2.14").get_trans_grid(), np.array([2.14])*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("(2, 3.5, 4)").get_trans_grid(), np.array([2, 3.5, 4])*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("[16, 2, 11]").get_trans_grid(), np.array([2, 11, 16])*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("1, 2.7, 2.6").get_trans_grid(), np.array([1, 2.6, 2.7])*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("linspace(2, 6, 5)").get_trans_grid(), np.linspace(2, 6, 5)*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("linspace (2, 6, 5)").get_trans_grid(), np.linspace(2, 6, 5)*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("linspace(2, 6)").get_trans_grid(), np.linspace(2, 6, 50)*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("range(2, 7, 2)").get_trans_grid(), np.arange(2, 7, 2)*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("range(2, 7)").get_trans_grid(), np.arange(2, 7)*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("arange(2, 7)").get_trans_grid(), np.arange(2, 7)*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("range(7)").get_trans_grid(), np.arange(7)*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("range(7.3)").get_trans_grid(), np.arange(7.3)*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("arange(7.3)").get_trans_grid(), np.arange(7.3)*NM2ANGSTROM)
#     # increments
#     assert np.allclose(TranslationParser("[5, 6.5, 7, 12]").get_increments(), np.array([5, 1.5, 0.5, 5])*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("[5, 12, 6.5, 7]").get_increments(), np.array([5, 1.5, 0.5, 5])*NM2ANGSTROM)
#     assert np.allclose(TranslationParser("[12, 5, 6.5, 7]").get_increments(), np.array([5, 1.5, 0.5, 5])*NM2ANGSTROM)
#     # sum of increments
#     assert np.allclose(TranslationParser("[2, 2.5, 2.77, 3.4]").sum_increments_from_first_radius(), 1.4*NM2ANGSTROM)
#     # hash
#     tp = TranslationParser("[2, 2.5, 2.77, 3.4]")
#     assert tp.grid_hash == 1368241250
#     tp2 = TranslationParser("range(2, 7, 2)")
#     assert tp2.grid_hash == 753285640
#     tp3 = TranslationParser("[2, 4, 6]")
#     assert tp3.grid_hash == 753285640
#     tp4 = TranslationParser("linspace(2, 6, 3)")
#     assert tp4.grid_hash == 753285640


if __name__ == "__main__":
    test_energy_reader()
    #test_trans_parser()
