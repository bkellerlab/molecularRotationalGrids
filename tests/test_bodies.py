from molgri.bodies import AbstractShape, ShapeSet, Cuboid, Molecule, MoleculeSet, join_shapes, Atom, AtomSet
import numpy as np


def test_translate_radially():
    # single object in 2D
    abs_s = AbstractShape(2)
    abs_s.translate((-2, 3))
    initial_position = abs_s.position.copy()
    assert np.allclose(abs_s.position, (-2, 3))
    initial_len = 3.605551275463989
    change = 1.5
    abs_s.translate_radially(change)
    end_len = initial_len + change
    end_position = initial_position * end_len / initial_len
    assert np.allclose(abs_s.position, end_position), "Objects not correctly translated for a positive radial change."
    # move backwards
    second_change = -7
    abs_s.translate_radially(second_change)
    second_end_len = end_len + second_change
    second_end_position = end_position * second_end_len / end_len
    assert np.allclose(abs_s.position, second_end_position), "Objects not correctly translated for a negative\
     radial change."
    # multiple objects in 3D
    c1 = Cuboid()
    c2 = Cuboid(5, 1, 3)
    shape_s = ShapeSet([c1, c2])
    shape_s.translate_objects_radially(5)
    for shape in shape_s.all_objects:
        assert np.allclose(shape.position, (0, 0, 5)), "Objects in ShapeSet not correctly translated from origin."
    shape_s.translate_objects_radially(-5)
    for shape in shape_s.all_objects:
        assert np.allclose(shape.position, (0, 0, 0)), "Objects in ShapeSet not correctly translated for a\
         negative radial change."
    shape_s.rotate_objects_about_body((0, 0, np.pi/6), method="euler_123")
    shape_s.translate_objects_radially(5)
    for shape in shape_s.all_objects:
        assert np.allclose(shape.position[2], 5*np.cos(np.pi/6), atol=1e-3), "Objects in ShapeSet not correctly translated from\
         origin with tilted basis."
    shape_s.translate_objects_radially(-5)
    for shape in shape_s.all_objects:
        assert np.allclose(shape.position, (0, 0, 0), atol=1e-3),"Objects in ShapeSet not correctly translated \
             with tilted basis in backward direction."
    # molecules
    centers = np.array([[0, 0, 0,],
                        [-0.5, 0.5, -1],
                        [0.3, -0.5, -1],
                        [0.3, 0.5, -1]])
    m = Molecule(["N", "H", "H", "H"], centers=centers)
    dist = 3
    initial_com = m.position.copy()
    # position at origin
    m.translate(-initial_com)
    m.translate_radially(dist)
    assert np.allclose(m.position[2], dist)
    for i, atom in enumerate(m.atoms):
        assert np.allclose(atom.position[2], centers[i][2]-initial_com[2]+dist)


def test_join_shapes():
    c1 = Cuboid()
    c2 = Cuboid(2, 4, 5)
    m1 = Molecule(["O"], np.array([[0, 0, 5]]), center_at_origin=False)
    # a list of simple objects
    joined_set1 = join_shapes([c1, c2, m1])
    assert joined_set1.all_objects[0] is c1
    assert joined_set1.all_objects[1] is c2
    assert joined_set1.all_objects[2] is m1
    assert joined_set1.all_objects[2].atoms[0].element.symbol == "O"
    assert np.allclose(joined_set1.all_objects[2].atoms[0].position, np.array([[0, 0, 5]]))
    assert np.allclose(joined_set1.all_objects[0].position, np.array([[0, 0, 0]]))
    # mixed objects and sets
    set1 = ShapeSet([c1, c1, c2])
    m2 = Molecule(["O", "H"], np.array([[0, 0, 5], [3, 2, 1]]), center_at_origin=False)
    set2 = MoleculeSet([m2, m1])
    joined_set2 = join_shapes([set1, m1, set2])
    assert len(joined_set2.all_objects) == 6
    assert np.all([c1, c1, c2, m1, m2, m1] == joined_set2.all_objects)
    # generate a molecule set
    joined_set3 = join_shapes([set2, m1], as_molecule_set=True)
    assert isinstance(joined_set3, MoleculeSet)
    assert len(joined_set3.all_objects) == 3
    assert np.all([m2, m1, m1] == joined_set3.all_objects)
    assert joined_set3.num_atoms == 2 + 1 + 1
    # generate an atom set
    a1 = Atom("H")
    a2 = Atom("O")
    joined_set4 = join_shapes([a1, a2], as_atom_set=True)
    assert isinstance(joined_set4, AtomSet)
    assert len(joined_set4.all_objects) == 2
    assert np.all([a1, a2] == joined_set4.all_objects)
    assert joined_set4.num_atoms == 2


if __name__ == '__main__':
    test_translate_radially()