import numpy as np
from scipy.constants import pi

from ..objects.abstractshape import AbstractShape


class Sphere(AbstractShape):

    def __init__(self, radius=1, color="red"):
        self.radius = radius
        super().__init__(3, color=color)

    def __str__(self):
        return f"{self.color} {type(self).__name__} at {self.position} with radius {self.radius}"

    def draw(self, axis, **kwargs):
        """
        Draw the sphere with given radius.

        Args:
            axis: ax on which the object should be drown
        """
        alpha = kwargs.pop("alpha", 0.5)
        shade = kwargs.pop("shade", True)
        u, v = np.mgrid[-pi:pi:20j,
                        -pi:pi:20j]
        x = self.radius * np.cos(u) * np.sin(v)
        y = self.radius * np.sin(u) * np.sin(v)
        z = self.radius * np.cos(v)
        surf = axis.plot_surface(x + self.position[0], y + self.position[1], z + self.position[2],
                                 color=self.color, alpha=alpha, rstride=1, cstride=1, shade=shade)
        super_obj = super().draw(axis, **kwargs)
        return [surf, *super_obj]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    from plotting.plotting_helper_functions import set_axes_equal

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    my_sphere = Sphere(radius=1)
    my_sphere.draw(ax, show_basis=True)
    my_sphere.translate(np.array([-3, 3, -4]))
    my_sphere.rotate_about_origin(np.array([pi/4, pi/6, pi/3]))
    my_sphere.draw(ax, show_basis=True)
    set_axes_equal(ax)
    plt.show()
    plt.close()
