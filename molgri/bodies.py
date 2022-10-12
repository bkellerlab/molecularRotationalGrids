"""
Represent and draw a point in 3D space with a corresponding body basis that can be rotated and translated.
Basis for all other objects that translate and rotate.
"""

from abc import ABC
from collections import defaultdict
from typing import TextIO

import numpy as np
from matplotlib import patches
from matplotlib.axis import Axis
from mendeleev import element
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import ArrayLike
from scipy.constants import pi
from matplotlib.text import Text
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation

from .constants import PM2NM
from .rotations import Rotation2D


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

    # noinspection PyUnusedLocal
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
        self.basis = result[:self.dimension]
        self.position = result[self.dimension:self.dimension+1]
        self.drawing_points = result[self.dimension+1:]
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


class Cylinder(AbstractShape):

    def __init__(self, radius=1, height=3, color="green"):

        self.radius = radius
        self.height = height
        self.num_points = 50
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


class Atom(Sphere):

    def __init__(self, atom_name: str, start_position: np.ndarray = np.array([0, 0, 0]), belongs_to=None,
                 gro_label: list = None):
        if gro_label is None:
            gro_label = atom_name
        self.element = element(atom_name)
        self.gro_label = gro_label
        self.belongs_to = belongs_to
        super().__init__(radius=self.element.atomic_radius*PM2NM, color=self.element.jmol_color)
        self.translate(start_position)

    def draw(self, axis, **kwargs):
        alpha = kwargs.pop("alpha", 1)
        return super().draw(axis, alpha=alpha, shade=False, **kwargs)


class Molecule(AbstractShape):

    def __init__(self, atom_names: list, centers: np.ndarray, connections: np.ndarray = None, center_at_origin=False,
                 gro_labels: list = None):
        if gro_labels is None:
            gro_labels = atom_names
        self.atoms = []      # saving the Atom objects for easy plotting and access to properties
        for i, atom_name in enumerate(atom_names):
            self.atoms.append(Atom(atom_name, start_position=centers[i], gro_label=gro_labels[i]))
        if connections is None:
            connections = np.diag([1]*len(self.atoms))
        self.connections = connections
        super().__init__(dimension=3)
        self.position = self._calc_center_of_mass()
        if center_at_origin:
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


class ShapeSet(object):

    def __init__(self, all_objects: list or AbstractShape, num_dim: int = 3, name: str = "ShapeSet",
                 time_step: int = 0):
        """
        An object joining several AbstractShape objects so they can be translated, rotated or drawn together.
        Using 'which' command, only a part of the collection can be manipulated.
        """
        self.num_dim = num_dim
        self.time_step = time_step
        self.name = name
        if type(all_objects) == AbstractShape:
            self.all_objects = [all_objects]
        else:
            self.all_objects = all_objects
        assert [type(x) == AbstractShape for x in self.all_objects], "All simulated objects must be type(AbstractShape)"
        assert [x.dimension == self.num_dim for x in self.all_objects], "All objects must have the same" \
                                                                        "number of dimensions as the set"

    def draw_objects(self, axis: Axis, which: list = None, **kwargs) -> list:
        """
        The only method that actually draws. All other methods should call this method to draw.

        Args:
            axis: axis on which to draw
            which: list of 1s and 0s as long as self.all_objects or None - which objects should be drawn. If None, all.
            **kwargs: forwarded to AbstractShape.draw(), e.g. show_basis = True
        """
        all_images = []
        which = self._select_which_objects(which)
        for i, obj in enumerate(self.all_objects):
            if which[i]:
                all_images.extend(obj.draw(axis, **kwargs))
        return all_images

    def _select_which_objects(self, which: list or None):
        """
        Helper function to create a filter which objects to use from self.all_objects. If which=None, use all.
        """
        if which is None:
            which = [1] * len(self.all_objects)
        else:
            assert len(which) == len(self.all_objects), "which needs to provide 1/0 for each object in self.all_objects"
            assert all([x in [1, 0] for x in which]), "which cannot have values other than 1 or 0"
        return which

    def _manipulate_once(self, which_action: str, vector: ArrayLike, which: list = None, method: str = "euler_123",
                         inverse: bool = False):
        """
        Selects appropriate which_action and forwards to the corresponding function.

        Args:
            which_action: options: ['translate', 'rotate_objects_about_origin', 'rotate_objects_about_body']
            vector: vector for translation or vector/float of rotational angles
            which: list of 1s and 0s as long as self.all_objects or None - which objects should be drawn. If None, all
            method: what method of rotation
        """
        for j, _ in enumerate(self.all_objects):
            if which[j]:
                if which_action == "translate":
                    self.all_objects[j].translate(vector)
                elif which_action == "rotate_about_body":
                    return self.all_objects[j].rotate_about_body(vector, method=method, inverse=inverse)
                elif which_action == "rotate_about_origin":
                    return self.all_objects[j].rotate_about_origin(vector, method=method, inverse=inverse)
                else:
                    raise NotImplementedError("Only actions translate, rotate_about_body and" +
                                              "rotate_abut_origin are implemented.")

    def _manipulate_objects(self, which_action: str, vector: ArrayLike, which: list = None,
                            num_steps: int = 1, method: str = "euler_123", inverse=False, **kwargs):
        """
        Helper method that implements the background of translate_objects, rotate_objects_about_origin and
        rotate_objects_about_body.

        Args:
            num_steps: in how many steps the movement should happen
            the rest of arguments forwarded to self._manipulate_once()
        """
        vector = np.array(vector)
        which = self._select_which_objects(which)
        # can be done in several steps for a better illustration in simulation
        for i in range(num_steps):
            # do it for all objects selected with which command
            self._manipulate_once(which_action=which_action, vector=vector/num_steps, which=which, method=method,
                                  inverse=inverse)
            self.time_step += 1

    def translate_objects(self, vector: ArrayLike, which: list = None, **kwargs):
        """
        Translate all or some of objects for vector. Can be drawn or not, in multiple steps or not. See
        self._manipulate_objects for details.
        """
        self._manipulate_objects("translate", vector, which=which, **kwargs)

    def rotate_objects_about_origin(self, vector: ArrayLike, which: list = None, **kwargs):
        """
        Rotate all or some of objects about origin for a set of angles. Can be drawn or not, in multiple steps or not.
        See self._manipulate_objects for details.
        """
        self._manipulate_objects("rotate_about_origin", vector, which=which, **kwargs)

    def rotate_objects_about_body(self, vector: ArrayLike, which: list = None, **kwargs):
        """
        Rotate all or some of objects about body for a set of angles. Can be drawn or not, in multiple steps or not.
        See self._manipulate_objects for details.
        """
        self._manipulate_objects("rotate_about_body", vector, which=which, **kwargs)


class AtomSet(ShapeSet):

    def __init__(self, all_objects: list or AbstractShape, file_gro: TextIO = None,
                 file_xyz: TextIO = None, **kwargs):
        super().__init__(all_objects, **kwargs)
        assert [type(x) == Atom for x in self.all_objects], "All simulated objects must be Atoms"
        self.belongings = self._group_by_belonging()
        self.file_gro = file_gro
        self.file_xyz = file_xyz
        self.num_atoms = len(self.all_objects)

    def _group_by_belonging(self) -> dict:
        try:
            belonging_dic = defaultdict(list)
            for one_object in self.all_objects:
                belonging_dic[one_object.belongs_to].append(one_object)
            belonging_dic = dict(belonging_dic)
            return belonging_dic
        # this happens when initializing MoleculeSet
        except AttributeError:
            return dict()

    def _select_which_objects(self, which: list or None):
        """
        Helper function to create a filter which objects to use from self.all_objects. If which=None, use all.
        """
        if which is None:
            which = [1] * len(self.all_objects)
        elif which in list(self.belongings.keys()):
            return [x.belongs_to == which for x in self.all_objects]
        else:
            assert len(which) == len(self.all_objects), "which needs to provide 1/0 for each object in self.all_objects"
            assert all([x in [1, 0] for x in which]), "which cannot have values other than 1 or 0"
        return which

    def _save_atom_lines_gro(self, residue: str = "SOL", atom_num=1, residue_num=1):
        num_atom = atom_num
        num_molecule = residue_num
        for molecule_name, molecule_atoms in self.belongings.items():
            hydrogen_counter = 1
            for atom in molecule_atoms:
                pos_nm = atom.position
                if atom.element.symbol == "O":
                    name = "OW"
                elif atom.element.symbol == "H":
                    name = "HW" + str(hydrogen_counter)
                    hydrogen_counter += 1
                else:
                    name = atom.element.symbol
                self.file_gro.write(f"{num_molecule:5}{residue:5}{name:>5}{num_atom:5}{pos_nm[0]:8.3f}{pos_nm[1]:8.3f}"
                                    f"{pos_nm[2]:8.3f}{0:8.4f}{0:8.4f}{0:8.4f}\n")
                num_atom += 1
            num_molecule += 1

    def _manipulate_objects(self, *args, save_to_gro=False, ** kwargs):
        super()._manipulate_objects(*args, **kwargs)


class MoleculeSet(AtomSet):

    def __init__(self, all_objects: list or AbstractShape, file_gro: TextIO = None, file_xyz: TextIO = None, **kwargs):
        super().__init__(all_objects, file_gro=file_gro, file_xyz=file_xyz, **kwargs)
        assert [type(x) == Molecule for x in self.all_objects], "All objects must be Molecules"
        # changing from AtomSet
        self.belongings = {f"molecule_{i}": key.atoms for i, key in enumerate(self.all_objects)}
        # changing from AtomSet
        self.num_atoms = sum([len(self.all_objects[i].atoms) for i in range(len(self.all_objects))])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    my_cylinder = Cylinder()
    my_cylinder.draw(ax, show_basis=False)
    my_cylinder.translate(np.array([-3, 3, -4]))
    my_cylinder.rotate_about_origin(np.array([pi / 4, pi / 6, pi / 3]))
    my_cylinder.draw(ax, show_basis=False)
    my_cylinder.rotate_about_body(np.array([pi / 4, pi / 6, pi / 3]))
    my_cylinder.draw(ax, show_basis=False)
    print(my_cylinder)
    plt.show()
