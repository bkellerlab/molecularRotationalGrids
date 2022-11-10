import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between_vectors(central_vec: np.ndarray, side_vector: np.ndarray) -> float:
    v1_u = unit_vector(central_vec)
    v2_u = unit_vector(side_vector)
    angle_vectors = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle_vectors
