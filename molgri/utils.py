import numpy as np


def norm_per_axis(array: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Returns the norm of the vector or along some axis of an array.
    Default behaviour: if axis not specified, normalise a 1D vector or normalise 2D array row-wise. If axis specified,
    axis=0 normalises column-wise and axis=1 row-wise.

    Args:
        array: numpy array containing a vector or a set of vectors that should be normalised - per default assuming
               every row in an array is a vector
        axis: optionally specify along which axis the normalisation should occur

    Returns:
        an array of the same shape as the input array where each value is the norm of the corresponding vector/row/column
    """
    if axis is None:
        if len(array.shape) > 1:
            axis = 1
        else:
            axis = 0
    my_norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return np.repeat(my_norm, array.shape[axis], axis=axis)


def normalise_vectors(array: np.ndarray, axis: int = None, length=1) -> np.ndarray:
    """
    Returns the unit vector of the vector or along some axis of an array.
    Default behaviour: if axis not specified, normalise a 1D vector or normalise 2D array row-wise. If axis specified,
    axis=0 normalises column-wise and axis=1 row-wise.

    Args:
        array: numpy array containing a vector or a set of vectors that should be normalised - per default assuming
               every row in an array is a vector
        axis: optionally specify along which axis the normalisation should occur
        length: desired new length for all vectors in the array

    Returns:
        an array of the same shape as the input array where vectors are normalised, now all have length 'length'
    """
    assert length > 0, "Length of a vector cannot be negative"
    my_norm = norm_per_axis(array=array, axis=axis)
    return length * np.divide(array, my_norm)


def angle_between_vectors(central_vec: np.ndarray, side_vector: np.ndarray) -> np.array:
    """
    Having two vectors or two arrays in which each row is a vector, calculate all angles between vectors.
    For arrays, returns an array giving results like those:

    ------------------------------------------------------------------------------------
    | angle(central_vec[0], side_vec[0])  | angle(central_vec[0], side_vec[1]) | ..... |
    | angle(central_vec[1], side_vec[0])  | angle(central_vec[1], side_vec[1]  | ..... |
    | ..................................  | .................................  | ..... |
    ------------------------------------------------------------------------------------

    Angle between vectors equals the distance between two points measured on a surface of an unit sphere!

    Args:
        central_vec: first vector or array of vectors
        side_vector: second vector or array of vectors

    Returns:

    """
    assert central_vec.shape[-1] == side_vector.shape[-1], f"Last components of shapes of both vectors are not equal:" \
                                                     f"{central_vec.shape[-1]}!={side_vector.shape[-1]}"
    v1_u = normalise_vectors(central_vec)
    v2_u = normalise_vectors(side_vector)
    angle_vectors = np.arccos(np.clip(np.dot(v1_u, v2_u.T), -1.0, 1.0))
    return angle_vectors


def dist_on_sphere(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    Same as dist_on_sphere, but accepts and returns arrays and should only be used for already unitary vectors.

    Args:
        vector1: vector shape (n1, d) or (d,)
        vector2: vector shape (n2, d) or (d,)

    Returns:
        an array the shape (n1, n2) containing distances between both sets of points on sphere
    """
    norm1 = norm_per_axis(vector1)
    norm2 = norm_per_axis(vector2)
    assert np.allclose(norm1, norm2), "Both vectors/arrays of vectors don't have the same norms!"
    angle = angle_between_vectors(vector1, vector2)
    return angle * norm1
