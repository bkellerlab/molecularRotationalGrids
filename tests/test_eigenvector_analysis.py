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


def test_vmdcreator():
    vmd_creator = VMDCreator("test", "index < 3", "index >= 3")

    my_eigenvec_array = [[0, 5, -5, 0.6, 7, 3],
                         [-17, 3, 0.3, -0.4, 0.03, -16],
                         [-4, 5, 6, 7, 10, -2.2]]

    names = ["eigenvector0.tga", "eigenvector1.tga", "eigenvector2.tga"]

    output = vmd_creator.prepare_eigenvector_script(np.array(my_eigenvec_array), plot_names=names, num_extremes=2)

    expected_output ="""mol modstyle 0 0 CPK
color Display Background white
axes location Off
mol color Name
mol representation CPK 1.000000 0.300000 12.000000 10.000000
mol selection all
mol material Opaque
mol modcolor 1 0 Type
display projection Orthographic
display shadows on
display ambientocclusion on
material add copy AOChalky
material change shininess Material22 0.000000



mol addrep 0
mol modselect 0 0 index < 3
mol modcolor 0 0 Type


mol addrep 0
mol modselect 1 0 index >= 3
mol drawframes 0 1 {2, 5}
mol modcolor 1 0 Type


mol addrep 0
mol modselect 2 0 index >= 3
mol modcolor 2 0 ColorID 0
mol drawframes 0 2 {3, 2}
mol addrep 0
mol modselect 3 0 index >= 3
mol modcolor 3 0 ColorID 1
mol drawframes 0 3 {6, 1}



mol addrep 0
mol modselect 4 0 index >= 3
mol modcolor 4 0 ColorID 0
mol drawframes 0 4 {4, 5}
mol addrep 0
mol modselect 5 0 index >= 3
mol modcolor 5 0 ColorID 1
mol drawframes 0 5 {6, 1}


mol showrep 0 1 0

mol showrep 0 2 0

mol showrep 0 3 0

mol showrep 0 4 0

mol showrep 0 5 0

mol showrep 0 6 0

mol showrep 0 7 0

mol showrep 0 8 0


mol showrep 0 0 1
mol showrep 0 0 1
mol showrep 0 1 1
render TachyonInternal eigenvector0.tga
mol showrep 0 0 0
mol showrep 0 1 0



mol showrep 0 0 1
mol showrep 0 2 1
mol showrep 0 3 1
render TachyonInternal eigenvector1.tga
mol showrep 0 2 0
mol showrep 0 3 0



mol showrep 0 0 1
mol showrep 0 4 1
mol showrep 0 5 1
render TachyonInternal eigenvector2.tga
mol showrep 0 4 0
mol showrep 0 5 0

quit"""
    print(output.strip()==expected_output)

if __name__ == "__main__":
    test_finding_indices()
    test_vmdcreator()