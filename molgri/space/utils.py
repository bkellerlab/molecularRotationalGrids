"""
Useful functions for rotations and vectors.

Functions that belong in this module perform simple conversions, normalisations or assertions that are useful at various
points in the molgri.space subpackage.
"""
from typing import Tuple

import numpy as np
from numpy._typing import ArrayLike, NDArray
from numpy.typing import NDArray, ArrayLike
from scipy.constants import pi
from scipy.spatial.transform import Rotation

from molgri.constants import UNIQUE_TOL


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


def hemisphere_quaternion_set(quaternions: NDArray, upper=True) -> NDArray:
    """
    Select only the "upper half"/"bottom half" of hyperspherical points (quaternions that may be repeating).
    How selection is done:
    for all points select either q or -q, depending which is in the right hemisphere

    Args:
        quaternions: array (N, 4), each row a quaternion
        upper: if True, select the upper hemisphere, that is, demand that the first non-zero coordinate is positive

    Returns:
        quaternions: array (M <= N, 4), each row a quaternion different from all other ones
    """
    # test input
    is_array_with_d_dim_r_rows_c_columns(quaternions, d=2, c=4)

    non_repeating_quaternions = []
    for projected_point in quaternions:
        for i in range(4):
            # if this if-sentence is True, the point is in the upper hemisphere
            if np.allclose(projected_point[:i], 0) and projected_point[i] > 0:
                # the point is selected
                if upper:
                    non_repeating_quaternions.append(projected_point)
                else:
                    non_repeating_quaternions.append(find_inverse_quaternion(projected_point))
                break
        # if the loop didn't break, the point was not in upper hemisphere
        else:
            if upper:
                non_repeating_quaternions.append(find_inverse_quaternion(projected_point))
            else:
                non_repeating_quaternions.append(projected_point)

    return np.array(non_repeating_quaternions)


def find_inverse_quaternion(q: NDArray) -> NDArray:
    """
    Inverse quaternion -q = (-q0, -q1, -q2, -q3) is the quaternion that represents the same rotation as q.

    Args:
        q: a quaternion of shape (4,) whose inverse is needed

    Returns:
        another quaternion of shape (4,) with all coordinates inversed
    """
    assert q.shape == (4,)
    return -q


def q_in_upper_sphere(q: NDArray) -> bool:
    """
    Determine whether q in the upper part of the (hyper)sphere. This will be true if the first non-zero element of
    the vector/quaternion is positive.

    The point of all zeros is defined to be in the bottom hemisphere.

    Args:
        q: a vector/quaternion to be tested

    Returns:

    """
    assert len(q.shape) == 1
    for i, q_i in enumerate(q):
        if np.allclose(q[:i], 0) and q[i] > 0:
            return True
    return False

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


def distance_between_quaternions(q1: NDArray, q2: NDArray) -> ArrayLike:
    """
    Calculate the distance between two unit quaternions or the pairwise distances between two arrays of unit
    quaternions. Quaternion distance is like hypersphere distance, but also considers double coverage.
    Args:
        q1 (): array either of shape (4,) or (N, 4), every row has unit length
        q2 (): array either of shape (4,) or (N, 4), every row has unit length

    Returns:
        Float or an array of shape (N,) containing distances between unit quaternions.
    """
    if q1.shape == (4,) and q2.shape == (4,):
        theta = angle_between_vectors(q1, q2)
    elif q1.shape[1] == 4 and q2.shape[1] == 4 and q1.shape[0]==q2.shape[0]:
        theta = np.diagonal(angle_between_vectors(q1, q2))
    else:
        raise ValueError("Shape of quaternions not okay")
    # if the distance would be more than half hypersphere, use the smaller distance
    return np.where(theta > pi / 2, pi-theta, theta)


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


def k_is_a_row(my_array: NDArray, k: NDArray) -> bool:
    """
    Check if the row k occurs anywhere in the array up to floating point precision
    Args:
        my_array ():
        k ():
        atol ():
        rtol ():

    Returns:

    """
    is_array_with_d_dim_r_rows_c_columns(k, r=my_array.shape[1])
    return np.any(np.all(np.isclose(k, my_array), axis=1))


def which_row_is_k(my_array: NDArray, k: NDArray) -> ArrayLike:
    """
    returns all indices of rows in my_array that are equal (within floating point errors) to my_array.
    Args:
        my_array:
        k:

    Returns:

    """
    return np.nonzero(np.all(np.isclose(k, my_array), axis=1))[0]


def all_row_norms_equal_k(my_array: NDArray, k: float, atol: float = None, rtol: float = None) -> NDArray:
    """
    Same as all_row_norms_similar, but also test that the norm equals k.
    """
    my_norms = all_row_norms_similar(my_array=my_array, atol=atol, rtol=rtol)
    assert check_equality(my_norms, np.array(k), atol=atol, rtol=rtol), "The norms are not equal to k"
    return my_norms


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


def all_rows_unique(my_array: NDArray, tol: int = UNIQUE_TOL):
    """
    Check if all rows of the array are unique up to tol number of decimal places.
    """
    my_unique = np.unique(my_array.round(tol), axis=0)
    difference = np.abs(len(my_array) - len(my_unique))
    assert len(my_array) == len(my_unique), f"{difference} elements of an array are not unique up to tolerance."


def quaternion_in_array(quat: NDArray, quat_array: NDArray) -> bool:
    """
    Check if a quaternion q or its equivalent complement -q is present in the quaternion array quat_array.
    """
    quat1 = quat[np.newaxis, :]
    for quat2 in quat_array:
        if two_sets_of_quaternions_equal(quat1, quat2[np.newaxis, :]):
            return True
    return False


def two_sets_of_quaternions_equal(quat1: NDArray, quat2: NDArray) -> bool:
    """
    This test is necessary because for quaternions, q and -q represent the same rotation. You therefore cannot simply
    use np.allclose to check if two sets of rotations represented with quaternions are the same. This function checks
    if all rows of two arrays are the same up to a flipped sign.
    """
    assert quat1.shape == quat2.shape
    assert quat1.shape[1] == 4
    # quaternions are the same if they are equal up to a +- sign
    # I have checked this fact and it is mathematically correct
    for q1, q2 in zip(quat1, quat2):
        if not (np.allclose(q1, q2) or np.allclose(q1, find_inverse_quaternion(q2))):
            return False
    return True


def norm_per_axis(array: NDArray, axis: int = None) -> NDArray:
    """
    Returns the norm of the vector or along some axis of an array.
    Default behaviour: if axis not specified, normalise a 1D vector or normalise 2D array row-wise. If axis specified,
    axis=0 normalises column-wise and axis=1 row-wise.

    Args:
        array: numpy array containing a vector or a set of vectors that should be normalised - per default assuming
               every row in an array is a vector
        axis: optionally specify along which axis the normalisation should occur

    Returns:
        an array of the same shape as the input array where each value is the norm of the corresponding
        vector/row/column
    """
    if axis is None:
        if len(array.shape) > 1:
            axis = 1
        else:
            axis = 0
    my_norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return np.repeat(my_norm, array.shape[axis], axis=axis)


def points4D_2_8cells(point_array:NDArray) -> tuple:
    """
    Take one array of shape (N, 4) where every row is (q_i0, q_i1, q_i2, q_i3) as input and return a list of 8
    sublists, each sublist containing a (possibly zero) number of 3D coordinates and arranged so:
    - sublist 0 has all points with q_i0 <= 0
    - sublist 1 has all points with q_i1 <= 0
    ...
    - sublist 6 has all points with q_i2 >= 0
    - sublist 7 has all points with q_i3 >= 0

    The sublist-defining coordinate is removed so that contents of sublists are 3-dimensional.
    Args:
        point_array ():

    Returns:

    """
    assert is_array_with_d_dim_r_rows_c_columns(point_array, d=2, c=4)
    output = [[] for _ in range(8)]
    output_indices = [[] for _ in range(8)] # needs to be created since we delete one component
    for coo in range(4):
        # skipping the coo column in each row
        without_coo = np.concatenate((point_array[:, :coo], point_array[:, coo+1:]), axis=1)
        neg_rows = np.nonzero(point_array[:, coo]<0)[0]
        output_indices[coo].extend(neg_rows)
        output[coo].extend(without_coo[neg_rows].tolist())
        pos_rows = np.nonzero(point_array[:, coo] >= 0)[0]
        output[coo+4].extend(without_coo[pos_rows].tolist())
        output_indices[coo+4].extend(pos_rows)
    return output, output_indices


def exclude_nans(arr: NDArray) -> NDArray:
    """
    Take an input array and output another array in which all rows and columns that exclusively consist of nans are
    removed.

    Args:
        arr (NDArray): a 2D numpy array that may or may not have some rows/columns that ar all nans

    Returns:
        a 2D numpy array that has no rows/columns that ar all nans
    """
    valid_rows = np.all(~np.isnan(arr), axis=1)
    valid_columns = np.all(~np.isnan(arr), axis=0)

    # Extract the valid rows and columns from the original array
    extracted_arr = arr[valid_rows, :]
    extracted_arr = extracted_arr[:, valid_columns]
    return extracted_arr


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

    all_row_norms_equal_k(points, 1), "Not points on a unit sphere"
    vector_center = normalise_vectors(np.average(points, axis=0))
    N = len(points)
    # angle between first point, center point, and each additional point
    alpha = np.zeros(N)  # initialize array
    vector_co = points[0] - vector_center
    for i in range(1, N):
        alpha_candidate = _get_alpha_with_spherical_cosine_law(vector_center, points[0], points[i])
        #print(alpha_candidate)
        #alpha_candidate = np.arccos(np.dot(vector_co, (points[index_center]-vector_center)))
        if is_ccw(points[0], vector_center, points[i]):
            alpha[i] = alpha_candidate
        else:
            alpha[i] = 2*pi - alpha_candidate
    #print(alpha)
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


# def _get_central_angles(vertices, center = None) -> NDArray:
#     """
#     Vertices must be sorted counter-clockwise. Calculate an array of central angles describing the angle one_vertex
#     --- center --- the following vertex.
#
#     Args:
#         vertices ():
#         center ():
#
#     Returns:
#
#     """
#     if center is None:
#         center = normalise_vectors(np.average(vertices, axis=0))
#     N = len(vertices)
#     # angle between first point, center point, and each additional point
#     alpha = np.zeros(N)  # initialize array
#     for index_center in range(0, N):
#         alpha[index_center] = np.arccos(np.dot((vertices[index_center-1]-center), (vertices[index_center] - center)))
#     return alpha


def _get_alpha_with_spherical_cosine_law(A: NDArray, B: NDArray, C: NDArray):
    """
    A, B and C are points on a sphere that form a triangle, given as vectors in cartesian coordinates. We use
    the spherical law of cosines to obtain the angle at point A.
    """
    # check that they all have the same norm (are on the same sphere)
    assert np.allclose(np.linalg.norm(A), np.linalg.norm(B)) and np.allclose(np.linalg.norm(A), np.linalg.norm(C))
    # consider spherical triangle:
    A = normalise_vectors(A)
    B = normalise_vectors(B)
    C = normalise_vectors(C)
    # and lengths of the opposite sides a, b, c are
    a = dist_on_sphere(B, C)[0]
    b = dist_on_sphere(C, A)[0]
    c = dist_on_sphere(A, B)[0]
    # using cosine law on spheres (need rounding so we don't numerically get over/under the range of arccos):
    alpha = np.arccos(np.round((np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c)), 7))
    #print(a, b, c, (np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c)), alpha)
    return alpha


def exact_area_of_spherical_polygon(vertices: NDArray, r: float = 1) -> float:
    """
    Use the formula S=[\sum{\theta_i} - (n-2)\pindex_center]r^2 (Todhunter, I. (1886). Spherical Trigonometry), to calculate the
    area of a spherical polygon with given
    vertices. In the formula, r, is the radius of the sphere, n the number of polygon vertices, theta_i-s are the radian
    angles of a spherical polygon.

    Args:
        vertices (NDArray): A (m, 3) array of points on a unit 2-sphere. Points must be ordered counter-clockwise and
        unique.

    Returns:

    """
    n = len(vertices)
    thetas = []
    for i in range(n):
        # using cosine law on spheres:
        theta = _get_alpha_with_spherical_cosine_law(vertices[i], vertices[i-1], vertices[(i + 1) % n])
        thetas.append(theta)
    area = (np.sum(thetas) - (n-2)*pi) * r**2
    # chose the smaller of two possible spherical polygons described by these vertices
    if area > 2 * pi * r**2:
        area = 4 * pi * r**2 - area
    assert area >= 0, f"Area cannot be negative!"
    return area


def k_argmin_in_array(my_array: NDArray, k: int):
    """
    Of all the values in the array, find the indices of the k smallest values.

    Args:
        my_array (): array in which to search
        k (): number of results

    Returns:
        k indices indicating smallest item, second smallest etc
    """

    idx = np.argpartition(my_array, k)
    return idx[:k]


def k_argmax_in_array(my_array: NDArray, k: int):
    """
    Of all the values in the array, find the indices of the k largest values.

    Args:
        my_array (): array in which to search
        k (): number of results

    Returns:
        k indices indicating larges item, second largest etc
    """
    return np.argpartition(my_array, -k)[-k:]
