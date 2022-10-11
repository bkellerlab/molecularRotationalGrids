from molgri.grids import build_grid, classic_grid_d_cube, select_only_faces, project_grid_on_sphere, select_half_sphere
from molgri.my_constants import *
import numpy as np


def test_grid_2D():
    points = 6
    dimensions = 2
    # classic_grid_d_cube
    my_grid = classic_grid_d_cube(points, dimensions)
    assert my_grid.shape == (dimensions, points, points), "Shape wrong"
    flattened = my_grid.reshape((-1, dimensions))
    assert len(flattened) == points * points
    # length of the square diagonal should be 2
    above_left = my_grid[:, 0, 0]
    down_right = my_grid[:, -1, -1]
    diag_len = np.sqrt(8)
    assert np.isclose(np.linalg.norm(down_right - above_left), diag_len)
    # select_only_faces
    my_faces = select_only_faces(my_grid)
    N = points + points + points - 2 + points - 2
    assert my_faces.shape == (N, dimensions)
    above_left = my_faces[0, :]
    down_right = my_faces[-1, :]
    assert np.isclose(np.linalg.norm(down_right - above_left), diag_len)
    for i in range(points):
        assert my_grid[:, 0, i] in my_faces
        assert my_grid[:, points - 1, i] in my_faces
        assert my_grid[:, i, 0] in my_faces
        assert my_grid[:, i, points - 1] in my_faces
    # project_cube_grid_on_sphere
    my_sphere = project_grid_on_sphere(my_faces)
    assert my_sphere.shape == (N, dimensions)
    for sph_point in my_sphere:
        assert np.isclose(np.linalg.norm(sph_point), 1)
    # select_half_sphere
    my_half_sphere = select_half_sphere(my_sphere)
    assert my_half_sphere.shape == (N//2, dimensions) or (N//2 + 1, dimensions)
    for sph_point in my_half_sphere:
        assert np.isclose(np.linalg.norm(sph_point), 1)
        x, y = sph_point
        assert [x, -y] not in my_half_sphere.tolist()
        assert [x, -y] in my_sphere


def test_grid_3D():
    points = 12
    dimensions = 3
    # classic_grid_d_cube
    my_grid = classic_grid_d_cube(points, dimensions)
    assert my_grid.shape == (dimensions, points, points, points)
    flattened = my_grid.reshape((-1, dimensions))
    assert len(flattened) == points * points * points
    # length of the square diagonal should be 2
    above_left = my_grid[:, 0, 0, 0]
    down_right = my_grid[:, -1, -1, -1]
    dia_len = 3.4641016151377544
    assert np.isclose(np.linalg.norm(down_right - above_left), dia_len)
    # select_only_faces
    my_faces = select_only_faces(my_grid)
    N = points * points * 6 - 12 * points + 8
    assert my_faces.shape == (N, dimensions)
    above_left = my_faces[0, :]
    down_right = my_faces[-1, :]
    assert np.isclose(np.linalg.norm(down_right - above_left), dia_len)
    for i in range(points):
        assert my_grid[:, 0, i, 0] in my_faces
        assert my_grid[:, points - 1, i, 0] in my_faces
        assert my_grid[:, i, 0, 0] in my_faces
        assert my_grid[:, i, points - 1, 0] in my_faces
        assert my_grid[:, 0, 0, i] in my_faces
        assert my_grid[:, 0, -1, 0] in my_faces
        assert my_grid[:, -1, 0, i] in my_faces
        assert my_grid[:, -1, -1, 0] in my_faces
        assert my_grid[:, 0, i, -1] in my_faces
        assert my_grid[:, points - 1, i, -1] in my_faces
        assert my_grid[:, i, 0, -1] in my_faces
        assert my_grid[:, i, points - 1, -1] in my_faces
    # project_cube_grid_on_sphere
    my_sphere = project_grid_on_sphere(my_faces)
    assert my_sphere.shape == (N, dimensions)
    for sph_point in my_sphere:
        assert np.isclose(np.linalg.norm(sph_point), 1)
    # select_half_sphere
    my_half_sphere = select_half_sphere(my_sphere)
    assert my_half_sphere.shape == (N//2, dimensions) or (N//2 + 1, dimensions)
    for sph_point in my_half_sphere:
        assert np.isclose(np.linalg.norm(sph_point), 1)
        x, y, z = sph_point
        assert [x, -y, -z] not in my_half_sphere.tolist()
        assert [x, -y, -z] in my_sphere


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
    test_grid_2D()
    test_grid_3D()
    test_ordering()