"""
Represent and draw a point in 3D space with a corresponding body basis that can be rotated and translated.
"""

from abc import ABC
import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import pi
from matplotlib.text import Text
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation

from ..rotations.rotation_2D import Rotation2D


class AbstractShape(ABC):

    def __init__(self, dimension: int, drawing_points: np.ndarray = None, color="black"):
        """
        A shape is always created at origin. It can be represented by its basis in 2D or 3D.
        Its position and angles can later be changed with translations and rotation.

        Args:
            dimension: 2 or 3, how many dimensions in space does the object have.
        """
        assert dimension in [2, 3], "Dimension can only be 2 or 3"
        self.dimension = dimension
        # self.position and self.angles represent the internal body coordination system
        # (x, y, z) or (x, y)
        self.position = np.zeros(self.dimension, dtype=float)
        # basis vectors
        self.basis = np.eye(self.dimension, dtype=float)
        # points used for drawing of shape (num_points, self.dimensions)
        if drawing_points is None:
            self.drawing_points = np.zeros((1, self.dimension), dtype=float)
        else:
            self.drawing_points = drawing_points
        assert self.drawing_points.shape[1] in [2, 3], "drawing_points must have shape (num_points, self.dimensions)"
        self.color = color
        self.initial_state = (self.basis.copy(), self.position.copy(), self.drawing_points.copy())

    def __str__(self):
        return f"{self.color} {type(self).__name__} at {self.position}"

    def translate(self, vector: ArrayLike):
        """
        Move the position of the point by a given vector. This method can be appended by subclasses.
        The basis does not change (it is always drawn from the current self.position).

        Args:
            vector: array of shape (3,) or (2,) depending on self.dimension
        """
        vector = np.array(vector)
        assert len(vector) == self.position.shape[0], "Dimensions of the object space and vector space do not align."
        self.position += vector
        self.drawing_points += np.hstack(vector)

    def _rotate(self, angles: ArrayLike, method: str, **kwargs) -> Rotation:
        """
        Helper function to initialize the 3D Rotation object from the scipy module
        Args:
            angles: a list/array or number representing rotations in radians
            method: the type of rotation description: 'euler_123', 'euler_313', 'simple_3D_x', 'simple_3D_y'
                    or 'simple_3D_z'
        Returns:
            Rotation object
        """
        dict_methods = {"euler_123": "ZYX", "euler_313": "ZXZ", "simple_3D_x": "X", "simple_3D_y": "Y",
                        "simple_3D_z": "Z"}
        if method in dict_methods.keys():
            return Rotation.from_euler(dict_methods[method], angles)
        elif method == "quaternion":
            return Rotation.from_quat(angles)
        else:
            raise NotImplementedError(f"Method {method} is unknown or not implemented.")

    def from_123_to_313(self):
        """
        Helper function to switch from the Euler 123 representation to the 313 representation.
        """
        matrix_123_313 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.basis = (matrix_123_313 @ self.basis.T).T
        self.position = matrix_123_313 @ self.position
        self.drawing_points = (matrix_123_313 @ self.drawing_points.T).T

    def from_313_to_123(self):
        """
        Helper function to switch from the Euler 313 representation to the 123 representation.
        """
        matrix_313_123 = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.basis = (matrix_313_123 @ self.basis.T).T
        self.position = matrix_313_123 @ self.position
        self.drawing_points = (matrix_313_123 @ self.drawing_points.T).T

    def rotate_about_origin(self, angles: ArrayLike, method="euler_123", inverse: bool = False):
        """
        Rotate the object by angles given around the three coordinate axes. With respect to coordinate origin.

        Args:
            angles: array of shape (3,) or (2,) depending on self.dimension providing angles in radians
            method: 'euler_123', 'euler_313', 'simple_3D_x', 'simple_3D_y', 'simple_3D_z' or 'quaternion'
            inverse: True if the rotation should be inverted
        """
        if self.dimension == 2:
            rotation_mat = Rotation2D(angles)
        else:
            rotation_mat = self._rotate(angles, method)
        result = rotation_mat.apply(np.concatenate((self.basis, self.position[:, np.newaxis].T, self.drawing_points),
                                                   axis=0), inverse=inverse)
        self.basis, self.position, self.drawing_points = result[:self.dimension],\
                                                         result[self.dimension:self.dimension+1],\
                                                         result[self.dimension+1:]
        self.position = self.position.T.squeeze()
        return rotation_mat

    def rotate_about_body(self, angles: ArrayLike, method="euler_123", inverse: bool = False):
        """
        Rotate the object by angles given around the three coordinate axes. With respect to body coordinates.

        Args:
            angles: array of shape (3,) or (2,) depending on self.dimension providing angles in radians
            method:'euler_123', 'euler_313', 'simple_3D_x', 'simple_3D_y', 'simple_3D_z' or 'quaternion'
            inverse: True if the rotation should be inverted
        """
        if self.dimension == 2:
            rotation_mat = Rotation2D(angles)
        else:
            rotation_mat = self._rotate(angles, method)
        points_at_origin = self.drawing_points - np.hstack(self.position)
        result = rotation_mat.apply(np.concatenate((self.basis, points_at_origin), axis=0), inverse=inverse)
        self.basis, self.drawing_points = result[:self.dimension], result[self.dimension:]
        self.drawing_points += np.hstack(self.position)
        return rotation_mat

    def draw(self, axis: Axes, show_labels=False, show_basis=False, rotational_axis=None, rotational_center="origin")\
            -> list:
        """
        Draw the object in the given axis (2D or 3D). Possibly also draws the label with atom position, the
        body coordinate axes and/or the axis of rotation. Should be appended by subclasses for drawing the objects.

        Args:
            axis: ax on which the object should be drown
            show_labels: show a label with the position of the point
            show_basis: show the basis attached to the object
            rotational_axis: the 3D vector representing the axis around which to rotate
            rotational_center: 'origin' or 'body', where the rotational axis should be drawn.
        Returns:
            list of all objects to be plotted
        """
        to_return = []
        # draw rotational axis
        if np.any(rotational_axis):
            if rotational_center == 'body':
                origin = self.position
            elif rotational_center == 'origin':
                origin = np.array([0, 0, 0])
            else:
                raise ValueError(f"Unknown argument {rotational_center} for rotational center (try 'body' or 'origin')")
            normed = rotational_axis / np.linalg.norm(rotational_axis)
            axis_pic = axis.quiver(*origin, *normed, color="black", length=3, arrow_length_ratio=0.1)
            to_return.append(axis_pic)
        # draw label with the position of the body at the center of the body
        if show_labels:
            center_labels = self._draw_center_label(axis)
            to_return.append(center_labels)
        # draw the 2D/3D body axes
        if show_basis:
            basis_pic = self._draw_body_coordinates(axis)
            to_return.extend(basis_pic)
        return to_return

    def _draw_center_label(self, axis: Axes) -> Text:
        """
        Helper function for drawing the label of the object center.

        Args:
            axis: ax for drawing
        Returns:
            the Text object that should be plotted
        """
        if self.dimension == 2:
            label_text = axis.text(*self.position, s=f" ({self.position[0]}, {self.position[1]})")
        else:
            label_text = axis.text(*self.position, s=f" ({self.position[0]}, {self.position[1]}, {self.position[2]})")
        return label_text

    def _draw_body_coordinates(self, axis: Axes, draw_labels=True) -> list:
        """
        Helper function for drawing the coordinate system attached to the object.

        Args:
            axis: ax for drawing
            draw_labels: whether to label the basis with x_b, y_b, z_b
        Returns:
            list of all objects to be plotted (axes vectors, _create_labels)
        """
        if type(self) == AbstractShape:
            labels = [r"$x$", r"$y$", r"$z$"]
        else:
            labels = [r"$x_b$", r"$y_b$", r"$z_b$"]
        everything_to_plot = []
        for column in range(self.dimension):
            # basis is drawn with an origin at current position
            if self.dimension == 3:
                quiver = axis.quiver(*self.position, *self.basis[column, :], length=1, color="black")
            else:
                # length=1 in 2D does not exist for some reason, this is the workaround
                quiver = axis.quiver(*self.position, *self.basis[column, :],
                                     color="black", angles='xy', scale_units='xy', scale=1)
            # add the x, y, z _create_labels to the coordinate axes
            position_labels = np.vstack(self.position) + 1/2 * self.basis.T
            #position_labels += 0.2 * np.ones((self.dimension, self.dimension)) - 0.15 * np.eye(self.dimension)
            if draw_labels:
                text = axis.text(*position_labels[:, column], s=labels[column], fontsize=30)
                everything_to_plot.extend([quiver, text])
            else:
                everything_to_plot.extend([quiver])
        return everything_to_plot


class Point(AbstractShape):
    def draw(self, axis, **kwargs):
        axis.scatter(*self.position, color=self.color)
        super().draw(axis, **kwargs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    s = Point(dimension=3, color="green")
    s.draw(ax, show_basis=True)
    s.translate(np.array([2, 1, 3]))
    s.draw(ax, show_labels=True)
    for j in range(100):
        s.rotate_about_origin(np.array([pi/200, 0, 0]))
        if j % 20 == 0:
            s.draw(ax, show_basis=True)
        else:
            s.draw(ax)
    for j in range(100):
        s.rotate_about_origin(np.array([0, -pi/200, 0]))
        if j % 20 == 0:
            s.draw(ax, show_basis=True)
        else:
            s.draw(ax)
    for j in range(100):
        s.rotate_about_body(np.array([0, 0, -pi/200]))
        if j % 20 == 0:
            s.draw(ax, show_basis=True)
        else:
            s.draw(ax)
    plt.show()
