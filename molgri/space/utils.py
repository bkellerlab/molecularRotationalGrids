from typing import Tuple
import os

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial.transform import Rotation

####################################################################################################################
#                                      FILE HANDLING                                                               #
####################################################################################################################


def first_index_free_4_all(list_names, list_endings, list_paths, index_places: int = 4) -> int:
    """Similar to find_first_free_index, but you have a list of files (eg. trajectory, topology, log) -> all must
    be still free to use this specific index."""
    assert len(list_paths) == len(list_names) == len(list_endings)
    i = 0
    # the loop only makes sense till you use up all numbers that could be part of the name
    while i < 10**index_places:
        for name, ending, path in zip(list_names, list_endings, list_paths):
            # if any one of them exist, break
            if os.path.exists(format_name(file_path=path, file_name=name, num=i, places_num=index_places,
                                          suffix=ending)):
                i += 1
                break
        # if you did not break the loop, all of the indices are free
        else:
            return i
    raise FileNotFoundError(f"All file names with unique numbers {format(0, f'0{index_places}d')}-{10**index_places-1} "
                            f"are already used up!")


def paths_free_4_all(list_names, list_endings, list_paths, index_places: int = 4) -> tuple:
    num = first_index_free_4_all(list_names, list_endings, list_paths, index_places)
    result_paths = []
    for name, ending, path in zip(list_names, list_endings, list_paths):
        result = format_name(file_name=name, file_path=path, num=num, places_num=index_places, suffix=ending)
        result_paths.append(result)
    return tuple(result_paths)


def find_first_free_index(name: str, ending: str = None, index_places: int = 4, path: str = "") -> int:
    """
    Problem: you want to save a file with a certain name, eg. pics/PrettyFigure.png, but not overwrite any existing
    files.

    Solution: this function checks if the given file already exists and increases the index until a unique one is
    found. So for our example it may return the string pics/PrettyFigure_007.png

    If no such file exists yet, the return string will include as many zeros as defined by index_places

    Args:
        path: path to appropriate directory (if not current), ending with /, in the example 'pics/'
        name: name of the file, in the example 'PrettyFigure'
        ending: in the example, select 'png' (NO DOT)
        index_places: how many places are used for the number

    Returns:
        number - first free index (can be forwarded to format_name)
    """
    i = 0
    while os.path.exists(format_name(file_path=path, file_name=name, num=i, places_num=index_places, suffix=ending)):
        i += 1
    return i


def format_name(file_name: str, num: int = None, places_num: int = 4, suffix: str = None, file_path: str = ""):
    """

    Args:
        file_path: eg. output/
        file_name: eg my_file
        num: eg 17
        places_num: eg 4 -> num will be formatted as 0017
        suffix: ending of the file

    Returns:
        full path and file name in correct format
    """
    till_num = f"{file_path}{file_name}"
    if num is None:
        till_ending = till_num
    else:
        till_ending = f"{till_num}_{format(num, f'0{places_num}d')}"
    if suffix is None:
        return till_ending
    return f"{till_ending}.{suffix}"


####################################################################################################################
#                                      SPACE FUNCTIONS                                                             #
####################################################################################################################

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


def cart2sph(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Transform an individual 3D point from cartesian to spherical coordinates.

    Code obtained from Leon Wehrhan.
    """
    XsqPlusYsq = x ** 2 + y ** 2
    r = np.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))  # theta
    az = np.arctan2(y, x)  # phi
    return r, elev, az


def cart2sphA(pts: NDArray) -> NDArray:
    """
    Transform an array of shape (N, 3) in cartesian coordinates to an array of the same shape in spherical coordinates.
    Can be used to create a hammer projection plot. In this case, disregard column 0 of the output and plot columns
    1 and 2.

    Code obtained from Leon Wehrhan.
    """
    return np.array([cart2sph(x, y, z) for x, y, z in pts])


def standardise_quaternion_set(quaternions: NDArray, standard=np.array([1, 0, 0, 0])) -> NDArray:
    """
    Since quaternions double-cover rotations, standardise all quaternions in this array so that their dot product with
    the "standard" quaternion is positive. Return the quaternion array where some elements ma have been flipped to -q.

    Idea: method 2 described here: https://math.stackexchange.com/questions/3888504/component-wise-averaging-of-similar-quaternions-while-handling-quaternion-doubl

    Args:
        quaternions: array (N, 4), each row a quaternion
        standard: a single quaternion determining which half-hypersphere to use
    """
    assert len(quaternions.shape) == 2 and quaternions.shape[1] == 4, "Wrong shape for quaternion array"
    assert standard.shape == (4, ), "Wrong shape for standard quaternion"
    # Take the dot product of q1 with all subsequent quaternions qi, for 2≤i≤N, and negate any of the subsequent
    # quaternions whose dot product with qi is negative.
    dot_product = standard.dot(quaternions.T)
    standard_quat = quaternions.copy()
    standard_quat[np.where(dot_product < 0)] = -quaternions[np.where(dot_product < 0)]

    assert standard_quat.shape == quaternions.shape
    return standard_quat


def unique_quaternion_set(quaternions: NDArray) -> NDArray:
    """
    Standardise the quaternion set so that q and -q are converted to the same object. If now any repetitions exist,
    remove them from the array.

    Args:
        quaternions: array (N, 4), each row a quaternion

    Returns:
        quaternions: array (M <= N, 4), each row a quaternion different from all other ones
    """
    # standardizing is necessary to be able to use unique on quaternions
    nodes = standardise_quaternion_set(quaternions)
    # determine unique without sorting
    _, indices = np.unique(nodes, axis=0, return_index=True)
    quaternions = np.array([nodes[index] for index in sorted(indices)])
    return quaternions


def randomise_quaternion_set_signs(quaternions: NDArray) -> NDArray:
    """
    Return the set of the same quaternions up to the sign of each row, which is normalised.
    """
    assert len(quaternions.shape) == 2 and quaternions.shape[1] == 4, "Wrong shape for quaternion array"
    random_set = np.random.choice([-1, 1], size=(len(quaternions), 1))
    re_signed = quaternions * random_set
    assert re_signed.shape == quaternions.shape
    return re_signed


def random_sphere_points(n: int = 1000) -> NDArray:
    """
    Create n points that are truly randomly distributed across the sphere.

    Args:
        n: number of points

    Returns:
        an array of grid points, shape (n, 3)
    """
    z_vec = np.array([0, 0, 1])
    quats = random_quaternions(n)
    rot = Rotation.from_quat(quats)
    return rot.apply(z_vec)


def random_quaternions(n: int = 1000) -> NDArray:
    """
    Create n random quaternions

    Args:
        n: number of points

    Returns:
        an array of grid points, shape (n, 4)
    """
    result = np.zeros((n, 4))
    random_num = np.random.random((n, 3))
    result[:, 0] = np.sqrt(1 - random_num[:, 0]) * np.sin(2 * pi * random_num[:, 1])
    result[:, 1] = np.sqrt(1 - random_num[:, 0]) * np.cos(2 * pi * random_num[:, 1])
    result[:, 2] = np.sqrt(random_num[:, 0]) * np.sin(2 * pi * random_num[:, 2])
    result[:, 3] = np.sqrt(random_num[:, 0]) * np.cos(2 * pi * random_num[:, 2])
    assert result.shape[1] == 4
    return result