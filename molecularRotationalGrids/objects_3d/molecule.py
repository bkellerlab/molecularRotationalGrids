import numpy as np
from mendeleev import element
from scipy.constants import pi

from my_constants import *
from objects_3d.abstractshape import AbstractShape
from objects_3d.sphere import Sphere


class Atom(Sphere):

    def __init__(self, symbol: str, start_position: np.ndarray = np.array([0, 0, 0]), belongs_to=None):
        self.element = element(symbol)
        self.belongs_to = belongs_to
        super().__init__(radius=self.element.atomic_radius*PM2NM, color=self.element.jmol_color)
        self.translate(start_position)

    def draw(self, axis, **kwargs):
        alpha = kwargs.pop("alpha", 1)
        return super().draw(axis, alpha=alpha, shade=False, **kwargs)


class Molecule(AbstractShape):

    def __init__(self, atom_names: list, centers: np.ndarray, connections: np.ndarray = None):
        self.atoms = []      # saving the Atom objects for easy plotting and access to properties
        for i, atom_name in enumerate(atom_names):
            self.atoms.append(Atom(atom_name, start_position=centers[i]))
        if connections is None:
            connections = np.diag([1]*len(self.atoms))
        self.connections = connections
        super().__init__(dimension=3)
        self.position = self._calc_center_of_mass()
        self.translate(-self.position)

    def _calc_center_of_mass(self):
        total_mass = 0
        com = np.zeros(3)
        for i, atom in enumerate(self.atoms):
            total_mass += atom.element.atomic_weight
            com += atom.element.atomic_weight * self.atoms[i].position
        return com/total_mass

    def draw(self, axis, **kwargs):
        # currently only possible to draw single bonds
        plot_elements = []
        # draw bonds as connections of centers
        for i, line in enumerate(self.connections):
            for j, el in enumerate(line[:i]):
                if el:
                    plot_e = axis.plot(*zip(self.atoms[i].position, self.atoms[j].position), color="black", linewidth=2)
                    plot_elements.extend(plot_e)
        # draw atoms as solid spheres
        for atom in self.atoms:
            plot_e = atom.draw(axis, **kwargs)
            plot_elements.extend(plot_e)
        return plot_elements

    def translate(self, vector: np.ndarray):
        super().translate(vector)
        for atom in self.atoms:
            atom.translate(vector)

    def rotate_about_origin(self, angles: np.ndarray, **kwargs):
        super().rotate_about_origin(angles, **kwargs)
        for atom in self.atoms:
            atom.rotate_about_origin(angles, **kwargs)

    def rotate_about_body(self, angles: np.ndarray or float, **kwargs):
        super().rotate_about_body(angles, **kwargs)
        inverse = kwargs.pop("inverse", False)
        for atom in self.atoms:
            points_at_origin = atom.position - self.position
            rot = atom._rotate(angles, **kwargs)
            atom.position = rot.apply(points_at_origin, inverse=inverse) + self.position


class HF(Molecule):
    def __init__(self):
        distance = 91 * PM2NM
        super().__init__(atom_names=["F", "H"], centers=np.array([[0, 0, 0], [0, 0, distance]]),
                         connections=np.array([[0, 1], [1, 0]]))


class H2O(Molecule):
    def __init__(self):
        distance = 95.7 * PM2NM
        angle = np.deg2rad(104.50 - 90)
        dist1 = distance * np.cos(angle)
        dist2 = distance * np.sin(angle)
        super().__init__(atom_names=["O", "H", "H"],
                         centers=np.array([[0, 0, distance], [0, 0, 0], [0, dist1, distance + dist2]]),
                         connections=np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]))


class H2(Molecule):
    def __init__(self):
        distance = 74 * PM2NM
        super().__init__(atom_names=["H", "H"], centers=np.array([[0, 0, 0], [0, 0, distance]]),
                         connections=np.array([[0, 1], [1, 0]]))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    from plotting.plotting_helper_functions import set_axes_equal

    plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')

    my_HF = H2O()
    print(my_HF.position)
    my_HF.draw(ax)
    my_HF.translate(np.array([0.2, 0.3, 0.5]))
    print(np.linalg.norm(my_HF.atoms[0].position - my_HF.atoms[1].position))
    my_HF.rotate_about_body(np.array([pi/2, pi/2, pi/2]), method="euler_123")
    print(np.linalg.norm(my_HF.atoms[0].position - my_HF.atoms[1].position))
    my_HF.draw(ax)
    set_axes_equal(ax)
    plt.show()
