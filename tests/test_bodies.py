from molgri.bodies import AbstractShape, ShapeSet, Cuboid, Molecule
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
        assert np.allclose(shape.position[2], 5*np.cos(np.pi/6)), "Objects in ShapeSet not correctly translated from\
         origin with tilted basis."
    shape_s.translate_objects_radially(-5)
    for shape in shape_s.all_objects:
        assert np.allclose(shape.position, (0, 0, 0)),"Objects in ShapeSet not correctly translated \
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


if __name__ == '__main__':
    test_translate_radially()