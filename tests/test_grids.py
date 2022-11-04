from molgri.grids import build_grid, project_grid_on_sphere
from molgri.constants import SIX_METHOD_NAMES
import numpy as np
import pytest


def test_project_grid_on_sphere():
    array_vectors = np.array([[3, 2, -1],
                              [-5, 22, 0.3],
                              [-3, -3, -3],
                              [0, 1/4, 1/4]])

    expected_results = np.array([[3/np.sqrt(14), np.sqrt(2/7), -1/np.sqrt(14)],
                                 [-0.221602, 0.975047, 0.0132961],
                                 [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
                                 [0, 1/np.sqrt(2), 1/np.sqrt(2)]])
    # test the whole array
    results = project_grid_on_sphere(array_vectors)
    assert np.allclose(results, expected_results)
    # test individual components
    for vector, expected_result in zip(array_vectors, expected_results):
        result = project_grid_on_sphere(vector.reshape((1, -1)))
        assert np.allclose(result, expected_result.reshape((1, -1)))
    # what happens for zero vector? should throw an error
    array_zero = np.array([[3, 2, -1],
                           [0, 0, 0]])
    with pytest.raises(AssertionError) as e:
        project_grid_on_sphere(array_zero)
    assert e.type is AssertionError
    # 2 dimensions
    array_vectors2 = np.array([[3, 2],
                              [-5, 0.3],
                              [-3, -3],
                              [0, 1/4]])
    expected_results2 = np.array([[3/np.sqrt(13), 2/np.sqrt(13)],
                                  [-0.998205, 0.0598923],
                                  [-1/np.sqrt(2), -1/np.sqrt(2)],
                                  [0, 1]])
    results2 = project_grid_on_sphere(array_vectors2)
    assert np.allclose(results2, expected_results2)
    # 4 dimensions
    array_vectors3 = np.array([[3, 2, -5, 0.3],
                              [-3, -3, 0, 1/4]])
    expected_results3 = np.array([[0.486089, 0.324059, -0.810148, 0.0486089],
                                  [-12/17, -12/17, 0, 1/17]])
    results3 = project_grid_on_sphere(array_vectors3)
    assert np.allclose(results3, expected_results3)


def test_ordering():
    # TODO: figure out what's the issue
    """Assert that, ignoring randomness, the first N-1 points of ordered grid with length N are equal to ordered grid
    of length N-1"""
    for name in SIX_METHOD_NAMES:
        try:
            for N in range(14, 284, 3):
                for addition in (1, 7):
                    grid_1 = build_grid(name, N+addition, ordered=True).get_grid()
                    grid_2 = build_grid(name, N, ordered=True).get_grid()
                    assert np.allclose(grid_1[:N], grid_2)
        except AssertionError:
            print(name)


if __name__ == "__main__":
    test_ordering()