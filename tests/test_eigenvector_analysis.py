import numpy as np

from molgri.plotting.create_vmdlog import find_indices_of_largest_eigenvectors, VMDCreator


def test_finding_indices():
    """
    Make sure the function is able to identify largest pos/neg/abs values.
    """

    # without index list
    my_array_1 = np.array([-3, 2, 0.5, 16, 17, -22, -0.3, 0])
    abs_indices = find_indices_of_largest_eigenvectors(my_array_1, which="abs", add_one=False, num_extremes=2)
    assert set(abs_indices) == {5, 4}
    pos_indices = find_indices_of_largest_eigenvectors(my_array_1, which="pos", add_one=False, num_extremes=3)
    assert set(pos_indices) == {1, 3, 4}
    neg_indices = find_indices_of_largest_eigenvectors(my_array_1, which="neg", add_one=False, num_extremes=3)
    assert set(neg_indices) == {5, 0, 6}

    # with index list
    my_array_1 = np.array([-3, 2, 0.5, 16, 17, -22, -0.3, 0])
    index_list = [2, 5, np.nan, 0, 1, 4, np.nan, np.nan]

    abs_indices = find_indices_of_largest_eigenvectors(my_array_1, which="abs", add_one=False, num_extremes=2,
                                                       index_list=index_list)
    assert set(abs_indices) == {1, 4}
    pos_indices = find_indices_of_largest_eigenvectors(my_array_1, which="pos", add_one=False, num_extremes=3,
                                                       index_list=index_list)
    assert set(pos_indices) == {5, 0, 1}
    neg_indices = find_indices_of_largest_eigenvectors(my_array_1, which="neg", add_one=False, num_extremes=3,
                                                       index_list=index_list)
    for el in neg_indices:
        np.any(np.isclose(el, [4.0, 2.0, np.nan])) # because nan is a float this more annoying comparison type

def test_search_an_replace():

    my_str = "I have a keyword that I like very much."
    to_replace = ["keyword", "like"]
    new_values = ["problem", "hate"]

def test_vmdcreator():
    vmd_creator = VMDCreator("sqra_water_in_vacuum", "index < 3", "index >= 3")

    my_eigenvec_array = [[0, 5, -5, 0.6, 7, 3],
                         [-17, 3, 0.3, -0.4, 0.03, -16],
                         [-4, 5, 6, 7, 10, -2.2]]

    names = ["eigenvector0.tga", "eigenvector1.tga", "eigenvector2.tga"]

    output = vmd_creator.prepare_eigenvector_script(np.array(my_eigenvec_array), plot_names=names, num_extremes=2)
    print(output)

if __name__ == "__main__":
    test_finding_indices()
    test_vmdcreator()