"""
Create a cuboid that can be rotate_about_origind, translated or drawn.
"""
from scipy.constants import pi
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from objects_3d.abstractshape import AbstractShape


class Cuboid(AbstractShape):

    def __init__(self, len_x: float = 1, len_y: float = 2, len_z: float = 4, color: str = "blue"):
        """
        Cuboid is always a 3D object. It is created with a center at the origin.

        Args:
            len_x: length of side in x direction
            len_y: length of side in y direction
            len_z: length of side in z direction
        """

        self.side_lens = np.array([len_x, len_y, len_z], dtype=float)
        # (3, 8) array that saves the position of vertices
        # do not change the order or the entire class needs to be adapted!
        vertices = np.zeros((3, 8))
        vertices[0] = self.side_lens[0]/2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        vertices[1] = self.side_lens[1]/2 * np.array([-1, -1, 1, 1, -1, -1, 1, 1])
        vertices[2] = self.side_lens[2]/2 * np.array([-1, 1, -1, 1, -1, 1, -1, 1])
        super().__init__(dimension=3, drawing_points=vertices.T, color=color)

    def __str__(self):
        return f"{self.color} {type(self).__name__} at {self.position} with sides {self.side_lens}"

    def _create_all_faces(self):
        """
        Helper function to create a list of all rectangles that represent the six faces of a cuboid.
        """
        # numbers represent the order of vertices that create a face
        edge_sequences = ["0132", "4576", "6732", "5104", "5731", "4620"]
        all_faces = []
        for face_num in edge_sequences:
            # each row one of the corners of the rectangle
            rectangle = np.zeros((4, 3), dtype=float)
            for i, num in enumerate(face_num):
                rectangle[i] = self.drawing_points[int(num), :]
            all_faces.append(rectangle)
        return all_faces

    def draw(self, axis: Axes, show_vertices: bool = False, **kwargs):
        alpha = kwargs.pop("alpha", 0.5)
        cuboid = Poly3DCollection(self._create_all_faces(), color=self.color, alpha=alpha)
        axis.add_collection3d(cuboid)
        super_obj = super().draw(axis, **kwargs)
        # show dots at all vertices
        if show_vertices:
            scatter = axis.scatter(*self.drawing_points.T, color="black")
            return [cuboid, scatter, *super_obj]
        return [cuboid, *super_obj]


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

    my_cuboid = Cuboid(color="red")
    my_cuboid.draw(ax, show_basis=True, show_vertices=True)
    my_cuboid.translate(np.array([-3, 3, -4]))
    my_cuboid.rotate_about_origin(np.array([pi/4, pi/6, pi/3]))
    my_cuboid.color = "blue"
    my_cuboid.draw(ax, show_basis=True)
    my_cuboid.rotate_about_body(np.array([pi / 4, pi / 6, pi / 3]))
    my_cuboid.color = "pink"
    my_cuboid.draw(ax, show_basis=True)
    plt.show()
