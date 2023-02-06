
import numpy as np

from molgri.space.utils import normalise_vectors, standardise_quaternion_set, random_quaternions, \
    randomise_quaternion_set_signs
from molgri.assertions import two_sets_of_quaternions_equal


def test_normalising():
    # 1D default
    my_array1 = np.array([3, 15, -2])
    normalised = normalise_vectors(my_array1)
    assert np.isclose(np.linalg.norm(normalised), 1)

    # 1D
    my_array2 = np.array([3, 15, -2])
    length = 2.5
    normalised = normalise_vectors(my_array2, length=length)
    assert np.isclose(np.linalg.norm(normalised), length)

    # 2D default
    my_array3 = np.array([[3, 15, -2],
                         [7, -2.1, -0.1]])
    normalised3 = normalise_vectors(my_array3)
    assert normalised3.shape == my_array3.shape
    assert np.allclose(np.linalg.norm(normalised3, axis=1), 1)

    # 2D different length
    my_array4 = np.array([[3, 15, -2],
                         [7, -2.1, -0.1],
                          [-0.1, -0.1, -0.1],
                          [0, 0, 0.5]])
    length4 = 0.2
    normalised4 = normalise_vectors(my_array4, length=length4)
    assert normalised4.shape == my_array4.shape
    assert np.allclose(np.linalg.norm(normalised4, axis=1), length4)

    # 2D other direction
    my_array5 = np.random.random((3, 5))
    length5 = 1.99
    normalised5 = normalise_vectors(my_array5, axis=0, length=length5)
    assert normalised5.shape == my_array5.shape
    assert np.allclose(np.linalg.norm(normalised5, axis=0), length5)

    # 3D
    my_array6 = np.random.random((3, 2, 5))
    length6 = 3
    normalised6 = normalise_vectors(my_array6, axis=0, length=length6)
    assert normalised6.shape == my_array6.shape
    assert np.allclose(np.linalg.norm(normalised6, axis=0), length6)


def test_standardise_quaternion_set():
    quaternions = random_quaternions(500)
    base = random_quaternions(1).reshape((4,))
    standard_quat = standardise_quaternion_set(quaternions, base)
    assert two_sets_of_quaternions_equal(quaternions, standard_quat)
    assert np.all(base.dot(standard_quat.T)) >= 0


def test_randomise_quaternion_set_signs():
    quaternions = random_quaternions(500)
    rand_quaternions = randomise_quaternion_set_signs(quaternions)
    assert two_sets_of_quaternions_equal(quaternions, rand_quaternions)