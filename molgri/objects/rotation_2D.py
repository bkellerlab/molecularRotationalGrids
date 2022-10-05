"""
Extension of the simpy.spatial.transform.Rotation to work in 2D
"""
import numpy as np
from numpy.typing import ArrayLike


class Rotation2D:

    def __init__(self, alpha: float or ArrayLike):
        """
        Initializes 2D rotation matrix with an angle.

        Args:
            alpha: angle in radians
        """
        rot_matrix = np.array([[np.cos(alpha), np.sin(alpha)],
                               [-np.sin(alpha), np.cos(alpha)]])
        self.rot_matrix = rot_matrix

    def apply(self, vector_set: ArrayLike, inverse: bool = False) -> ArrayLike:
        """
        Applies 2D rotational matrix to a set of vectors of shape (N, 2)

        Args:
            vector_set: array (each column a vector) that should be rotated
            inverse: True if the rotation should be inverted

        Returns:
            rotated vector of shape (N, 2)
        """
        if inverse:
            inverted_mat = self.rot_matrix.T
            result = vector_set.dot(inverted_mat)
        else:
            result = vector_set.dot(self.rot_matrix)
        result = result.squeeze()
        return result
