import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes

from ..objects.abstractshape import AbstractShape


class Circle(AbstractShape):

    def __init__(self, radius: float = 1, color: str = "red"):
        self.radius = radius
        num_points = 100
        points = np.linspace(-2*pi, 2*pi, num_points)
        xs = self.radius * np.cos(points)
        ys = self.radius * np.sin(points)
        all_points = np.array([xs, ys])
        super().__init__(2, drawing_points=all_points.T, color=color)

    def __str__(self):
        return f"{self.color} {type(self).__name__} at {self.position} with radius {self.radius}"

    def draw(self, axis: Axes, **kwargs) -> list:
        """
        Draw a circle with given radius.

        Args:
            axis: ax on which the object should be drown
        """
        circle_patch = patches.Polygon(self.drawing_points, color=self.color, alpha=0.5)
        axis.add_patch(circle_patch)
        super_obj = super().draw(axis, **kwargs)
        return [circle_patch, *super_obj]


class Rectangle(AbstractShape):
    def __init__(self, len_x: float = 1, len_y: float = 2, color="blue"):
        self.side_lens = np.array([len_x, len_y])
        vertices = 1/2 * np.array([[-len_x, -len_y],
                                   [-len_x, len_y],
                                   [len_x, len_y],
                                   [len_x, -len_y]])
        super().__init__(2, drawing_points=vertices, color=color)

    def __str__(self):
        return f"{self.color} {type(self).__name__} at {self.position} with sides {self.side_lens}"

    def draw(self, axis: Axes, **kwargs) -> list:
        """
        Draw a rectangle

        Args:
            axis: ax on which the object should be drown
        """
        rectangle = patches.Polygon(self.drawing_points, color=self.color, alpha=0.5)
        axis.add_patch(rectangle)
        super_obj = super().draw(axis, **kwargs)
        return [rectangle, *super_obj]


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.xlabel('x')
    plt.ylabel('y')
    my_circle = Circle(radius=2, color="blue")
    my_circle.translate(np.array([2, 7]))
    my_circle.draw(ax, show_basis=True)
    my_circle.rotate_about_origin(np.array([pi/2]))
    my_circle.draw(ax, show_basis=True)
    my_rectangle = Rectangle(color="pink")
    my_rectangle.draw(ax, show_basis=True)
    my_rectangle.translate(np.array([-3, -2]))
    my_rectangle.translate(np.array([-1, 0]))
    my_rectangle.rotate_about_body(np.array([pi/2]))
    my_rectangle.draw(ax, show_basis=True)
    plt.show()
