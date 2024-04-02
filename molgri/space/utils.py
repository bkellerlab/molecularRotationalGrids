"""
Useful functions for rotations and vectors.

Functions that belong in this module perform simple conversions, normalisations or assertions that are useful at various
points in the molgri.space subpackage.
"""
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial.transform import Rotation


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

def normalise_vectors(array: NDArray, axis: int = None, length: float = 1) -> NDArray:
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
    assert length >= 0, "Length of a vector cannot be negative"
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

def sort_points_on_sphere_ccw(points: NDArray) -> NDArray:
    """
    Gets an array of points on a 2D sphere; returns an array of the same points, but ordered in a counter-clockwise
    manner.

    Args:
        points (NDArray): an array in which each row is a coordinate of a point on a unit sphere (2-sphere)

    Returns:
        the same array of points, but sorted in a counter-clockwise manner. The first point remains in first position.
    """

    def is_ccw(v_0, v_c, v_i):
        # checks if the smaller interior angle for the great circles connecting u-v and v-w is CCW (counter-clockwise)
        return (np.dot(np.cross(v_c - v_0, v_i - v_c), v_i) < 0)

    #all_row_norms_equal_k(points, 1), "Not points on a unit sphere"
    vector_center = normalise_vectors(np.average(points, axis=0))
    N = len(points)
    # angle between first point, center point, and each additional point
    alpha = np.zeros(N)  # initialize array
    vector_co = points[0] - vector_center
    for i in range(1, N):
        alpha_candidate = _get_alpha_with_spherical_cosine_law(vector_center, points[0], points[i])
        if is_ccw(points[0], vector_center, points[i]):
            alpha[i] = alpha_candidate
        else:
            alpha[i] = 2*pi - alpha_candidate
    assert np.all(alpha >= 0)

    output = points[np.argsort(alpha)]
    return output

def triangle_order_for_ccw_polygons(len_points: int) -> list:
    """
    You used the function sort_points_on_sphere_ccw to order points of a spherical polygon  in a ccw fashion. Now you
    want to plot these points with plot_trisurf. To do that perfectly, you need to specify triangles. These are for
    ordered points quite simply [[0, 1, 2], [0, 2, 3], [0, 3, 4], .... [0, len_points-2, len_points-1]]. This
    function creates this list of indices automatically.

    Returns:
        a list
    """
    output = []
    for i in range(1, len_points-1):
        suboutput = [0, i, i+1]
        output.append(suboutput)
    return output


def _get_alpha_with_spherical_cosine_law(A: NDArray, B: NDArray, C: NDArray):
    """
    A, B and C are points on a sphere that form a triangle, given as vectors in cartesian coordinates. We use
    the spherical law of cosines to obtain the angle at point A.
    """
    # check that they all have the same norm (are on the same sphere)
    #assert np.allclose(np.linalg.norm(A), np.linalg.norm(B)) and np.allclose(np.linalg.norm(A), np.linalg.norm(C))
    # consider spherical triangle:
    A = normalise_vectors(A)
    B = normalise_vectors(B)
    C = normalise_vectors(C)
    # and lengths of the opposite sides a, b, c are
    a = dist_on_sphere(B, C)
    b = dist_on_sphere(C, A)
    c = dist_on_sphere(A, B)
    # using cosine law on spheres (need rounding so we don't numerically get over/under the range of arccos):
    alpha = np.arccos(np.round((np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c)), 7))
    #print(a, b, c, (np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c)), alpha)
    return alpha[0]


def all_row_norms_equal_k(my_array: NDArray, k: float, atol: float = None, rtol: float = None) -> NDArray:
    """
    Same as all_row_norms_similar, but also test that the norm equals k.
    """
    my_norms = all_row_norms_similar(my_array=my_array, atol=atol, rtol=rtol)
    assert check_equality(my_norms, np.array(k), atol=atol, rtol=rtol), "The norms are not equal to k"
    return my_norms


def check_equality(arr1: NDArray, arr2: NDArray, atol: float = None, rtol: float = None) -> bool:
    """
    Use the numpy function np.allclose to compare two arrays and return True if they are all equal. This function
    is a wrapper where I can set my preferred absolute and relative tolerance
    """
    if atol is None:
        atol = 1e-8
    if rtol is None:
        rtol = 1e-5
    return np.allclose(arr1, arr2, atol=atol, rtol=rtol)


def all_row_norms_similar(my_array: NDArray, atol: float = None, rtol: float = None) -> NDArray:
    """
    Assert that in an 2D array each row has the same norm (up to the floating point tolerance).

    Returns:
        the array of norms in the same shape as my_array
    """
    is_array_with_d_dim_r_rows_c_columns(my_array, d=2)
    axis = 1
    all_norms = norm_per_axis(my_array, axis=axis)
    average_norm = np.average(all_norms)
    assert check_equality(all_norms, average_norm, atol=atol, rtol=rtol), "The norms of all rows are not equal"
    return all_norms

def is_array_with_d_dim_r_rows_c_columns(my_array: NDArray, d: int = None, r: int = None, c: int = None):
    """
    Assert that the object is an array. If you specify d, r, c, it will check if this number of dimensions, rows, and/or
    columns are present.
    """
    assert type(my_array) == np.ndarray, "The first argument is not an array"
    # only check if dimension if d specified
    if d is not None:
        assert len(my_array.shape) == d, f"The dimension of an array is not d: {len(my_array.shape)}=!={d}"
    if r is not None:
        assert my_array.shape[0] == r, f"The number of rows is not r: {my_array.shape[0]}=!={r}"
    if c is not None:
        assert my_array.shape[1] == c, f"The number of columns is not c: {my_array.shape[1]}=!={c}"
    return True