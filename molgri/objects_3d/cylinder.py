from scipy.constants import pi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from objects_3d.abstractshape import AbstractShape


class Cylinder(AbstractShape):

    def __init__(self, radius=1, height=3, color="green"):

        self.radius = radius
        self.height = height
        self.num_points = 50
        #us = np.linspace(-2 * pi * self.radius, 2 * pi * self.radius, self.num_points)
        us = np.linspace(-2 * pi, 2 * pi, self.num_points)
        zs = np.linspace(-self.height / 2, self.height, 2)
        us, zs = np.meshgrid(us, zs)
        xs = self.radius * np.cos(us)
        ys = self.radius * np.sin(us)
        bottom_circle = np.stack((xs[0], ys[0], zs[0]))
        upper_circle = np.stack((xs[1], ys[1], zs[1]))
        super().__init__(3, drawing_points=np.hstack((bottom_circle, upper_circle)).T, color=color)

    def __str__(self):
        return f"{self.color} {type(self).__name__} at {self.position} with radius {self.radius} & height {self.height}"

    def draw(self, axis, **kwargs):
        bottom_circle = self.drawing_points[:self.num_points, :]
        upper_circle = self.drawing_points[self.num_points:, :]
        surface = np.stack((bottom_circle, upper_circle), axis=1)
        # draw the top and bottom circle
        alpha = kwargs.pop("alpha", 0.5)
        surf1 = Poly3DCollection([bottom_circle, upper_circle], color=self.color, alpha=alpha)
        axis.add_collection3d(surf1)
        # draw the curved surface
        surf2 = axis.plot_surface(*surface.T, color=self.color, alpha=alpha)
        super_obj = super().draw(axis, **kwargs)
        return [surf1, surf2, *super_obj]


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

    my_cylinder = Cylinder()
    my_cylinder.draw(ax, show_basis=True)
    my_cylinder.translate(np.array([-3, 3, -4]))
    my_cylinder.rotate_about_origin(np.array([pi/4, pi/6, pi/3]))
    my_cylinder.draw(ax, show_basis=True)
    my_cylinder.rotate_about_body(np.array([pi / 4, pi / 6, pi / 3]))
    my_cylinder.draw(ax, show_basis=True)
    print(my_cylinder)
    plt.show()
