import os
import numpy as np

from molgri.space.utils import distance_between_quaternions, normalise_vectors, standardise_quaternion_set, \
    random_quaternions, \
    randomise_quaternion_set_signs
from molgri.logfiles import find_first_free_index
from molgri.assertions import two_sets_of_quaternions_equal


def test_find_first_free_index():
    # where no exists before
    my_file = "one_file"
    my_ending = "css"
    assert find_first_free_index(name=my_file, index_places=7, ending=my_ending) == 0
    # save some (and delete later)
    with open(f"{my_file}_001.{my_ending}", "w") as f:
        f.writelines("test example 001")
    assert find_first_free_index(name=my_file, index_places=3, ending=my_ending) == 0
    with open(f"{my_file}_000.{my_ending}", "w") as f:
        f.writelines("test example 000")
    assert find_first_free_index(name=my_file, index_places=3, ending=my_ending) == 2
    # delete them
    os.remove(f"{my_file}_000.{my_ending}")
    os.remove(f"{my_file}_001.{my_ending}")
    # without ending
    assert find_first_free_index(name=my_file, index_places=2) == 0
    with open(f"{my_file}_00", "w") as f:
        f.writelines("test example 00")
    assert find_first_free_index(name=my_file, index_places=2) == 1
    os.remove(f"{my_file}_00")


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


def test_distance_between_quaternions():
    # single quaternions
    q1 = np.array([1, 0, 0, 0])
    q2 = - q1
    # distance to itself and -itself is zero
    assert np.isclose(distance_between_quaternions(q1, q1), 0)
    assert np.isclose(distance_between_quaternions(q1, q2), 0)

    q3 = np.array([0.3, 0.9, 0, 1.7])
    q3 = q3/np.linalg.norm(q3)

    # distance is symmetric
    assert np.allclose(distance_between_quaternions(q1, q3), [distance_between_quaternions(q3, q1),
                        distance_between_quaternions(q2, q3), distance_between_quaternions(q3, q2)])

    # arrays of quaternions
    q_array1 = np.array([q1, q2, q3])
    q_array2 = np.array([q3, q3, q3])
    result_array = distance_between_quaternions(q_array1, q_array2)
    assert np.isclose(result_array[0], distance_between_quaternions(q1, q3))
    assert np.isclose(result_array[1], distance_between_quaternions(q2, q3))
    assert np.isclose(result_array[2], distance_between_quaternions(q3, q3))


if __name__ == "__main__":
    test_distance_between_quaternions()