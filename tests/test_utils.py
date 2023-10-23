import os
import numpy as np

from molgri.space.utils import (distance_between_quaternions, normalise_vectors, random_quaternions,
                                angle_between_vectors, dist_on_sphere)
from molgri.logfiles import find_first_free_index
from scipy.constants import pi



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


def test_unit_dist_on_sphere():
    # angle between two vectors must be equal to distance times the radius of sphere
    for i in range(1, 20):
        vec1 = i*normalise_vectors((np.random.random(size=(3,)) - 0.5) * 2)
        vec2 = i*normalise_vectors((np.random.random(size=(3,)) - 0.5) * 2)
        angle = i*angle_between_vectors(vec1, vec2)
        dist = dist_on_sphere(vec1, vec2)
        assert np.allclose(angle, dist)


def test_unit_vector():
    """Testing that the axis argument is used as expected."""
    # example of a 2D array
    test_array = np.array([[-3, -0.2, -7],
                           [6, 5, -1],
                           [2, 9, 9],
                           [-18, 0.6, -0.3]])
    # axis = 0 is column-wise
    normalised_0 = normalise_vectors(test_array, axis=0)
    assert normalised_0.shape == test_array.shape, "Shape shouldn't change when normalising vectors"
    assert np.allclose(normalised_0[:, 0], np.array([-0.155334, 0.310668, 0.103556, -0.932005]))
    assert np.allclose(normalised_0[:, 1], np.array([-0.0193892, 0.484729, 0.872513, 0.0581675]))
    assert np.allclose(normalised_0[:, 2], np.array([-0.611383, -0.0873404, 0.786064, -0.0262021]))
    # axis = 1 should be row-wise normalisation
    normalised_1 = normalise_vectors(test_array, axis=1)
    assert normalised_1.shape == test_array.shape, "Shape shouldn't change when normalising vectors"
    assert np.allclose(normalised_1[0], np.array([-0.393784, -0.0262522, -0.918828]))
    assert np.allclose(normalised_1[1], np.array([0.762001, 0.635001, -0.127]))
    assert np.allclose(normalised_1[2], np.array([0.15523, 0.698535, 0.698535]))
    assert np.allclose(normalised_1[3], np.array([-0.999306, 0.0333102, -0.0166551]))
    # same if axis not defined
    normalised_1 = normalise_vectors(test_array)
    assert normalised_1.shape == test_array.shape, "Shape shouldn't change when normalising vectors"
    assert np.allclose(normalised_1[0], np.array([-0.393784, -0.0262522, -0.918828]))
    # example of a simple 1D array
    test_simple_array = np.array([2, -7, -0.9])
    norm = normalise_vectors(test_simple_array)
    assert np.allclose(norm, np.array([0.272646, -0.95426, -0.122691]))
    # same if axis defined:
    norm = normalise_vectors(test_simple_array, axis=0)
    assert np.allclose(norm, np.array([0.272646, -0.95426, -0.122691]))


def test_angle_between():
    # in 2D
    vec1 = np.array([0, 1])
    vec2 = np.array([1, 0])
    assert np.isclose(angle_between_vectors(vec1, vec2), pi/2)
    # in 3D, non-unitary vectors
    vec3 = np.array([0, -1, 0])
    vec4 = np.array([0, 2, 0])
    assert np.isclose(angle_between_vectors(vec3, vec4), pi)
    subvec_50 = [2, 7, 3]
    subvec_51 = [-1, 2, 4]
    subvec_60  = [2, 1, 5]
    subvec_61 = [-8, -0.8, 9]
    vec5 = np.array([subvec_50, subvec_51])
    vec6 = np.array([subvec_60, subvec_61])
    norm_50 = np.linalg.norm(subvec_50)
    norm_51 = np.linalg.norm(subvec_51)
    norm_60 = np.linalg.norm(subvec_60)
    norm_61 = np.linalg.norm(subvec_61)
    manual_angle_00 = np.arccos(np.dot(subvec_50, subvec_60)/(norm_60*norm_50))
    manual_angle_01 = np.arccos(np.dot(subvec_50, subvec_61)/(norm_61*norm_50))
    manual_angle_10 = np.arccos(np.dot(subvec_51, subvec_60) / (norm_60 * norm_51))
    manual_angle_11 = np.arccos(np.dot(subvec_51, subvec_61) / (norm_61 * norm_51))
    assert np.isclose(angle_between_vectors(vec5, vec6)[0, 0], manual_angle_00)
    assert np.isclose(angle_between_vectors(vec5, vec6)[0, 1], manual_angle_01)
    assert np.isclose(angle_between_vectors(vec5, vec6)[1, 0], manual_angle_10)
    assert np.isclose(angle_between_vectors(vec5, vec6)[1, 1], manual_angle_11)



if __name__ == "__main__":
    test_find_first_free_index()
    test_normalising()
    test_distance_between_quaternions()
    test_unit_dist_on_sphere()
    test_unit_vector()
    test_angle_between()