from molgri.rotations import Rotation2D

from scipy.constants import pi
import numpy as np


def test_rotation_2D():
    rot = Rotation2D(pi/6)
    expected_matrix = np.array([[np.sqrt(3)/2, -1/2], [1/2, np.sqrt(3)/2]])
    assert np.allclose(rot.rot_matrix, expected_matrix)